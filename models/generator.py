import numpy as np
import torch
import torch.nn as nn
from transformers import BertConfig
from transformers import AlbertModel
import torch.nn.functional as F
import math
from utils.constant import n_slot
from models.IMOR import MultiRelationalGCN
from models.grace import Model, Encoder
import random
import heapq
from scipy import spatial


class Generator(nn.Module):
    def __init__(self, args, n_op, n_domain, update_id, ans_vocab, slot_mm, turn=2, turn_id=30006):
        super(Generator, self).__init__()
        bert_config = BertConfig.from_pretrained(args.model_name_or_path + "config.json")
        args.slot_size = n_slot
        args.ans_size = 200
        args.hidden_size = bert_config.hidden_size
        args.n_slot = n_slot
        self.n_slot = n_slot
        self.args = args
        self.slot_mm = slot_mm
        self.turn = turn
        self.turn_id = turn_id
        self.albert = AlbertModel.from_pretrained(args.model_name_or_path + "pytorch_model.bin", config=bert_config)
        self.albert.resize_token_embeddings(args.vocab_size)
        self.input_drop = nn.Dropout(p=0.5)
        smask = ans_vocab.sum(dim=-1).eq(0).long()
        smask = slot_mm.long().mm(smask)
        self.slot_mm = nn.Parameter(slot_mm, requires_grad=False)
        self.slot_ans_mask = nn.Parameter(smask, requires_grad=False)
        self.ans_vocab = nn.Parameter(torch.FloatTensor(ans_vocab.size(0), ans_vocab.size(1), args.hidden_size),
                                      requires_grad=True)
        self.max_ans_size = ans_vocab.size(-1)
        self.slot_ans_size = ans_vocab.size(1)
        self.eslots = ans_vocab.size(0)
        self.ans_bias = nn.Parameter(torch.FloatTensor(ans_vocab.size(0), ans_vocab.size(1), 1), requires_grad=True)
        self.pos_weight = nn.Parameter(torch.FloatTensor([1]), requires_grad=True)
        self.pos_bias = nn.Parameter(torch.FloatTensor([0]), requires_grad=True)
        self.hidden_size = bert_config.hidden_size
        self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)
        self.action_cls = nn.Linear(self.hidden_size, n_op)
        self.related_cls = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.domain_cls = nn.Linear(self.hidden_size, n_domain)
        self.n_op = n_op
        self.n_domain = n_domain
        self.update_id = update_id
        self.W_Q = nn.Linear(self.hidden_size, self.hidden_size)
        self.start_output = nn.Linear(self.hidden_size, self.hidden_size)
        self.end_output = nn.Linear(self.hidden_size, self.hidden_size)
        self.attention_key = nn.Linear(self.hidden_size, self.hidden_size)
        self.attention_query = nn.Linear(self.hidden_size, self.hidden_size)
        self.slot_w = nn.Linear(self.hidden_size, 1)
        self.history_w = nn.Linear(self.hidden_size, 1)
        self.imor_w = nn.Linear(self.hidden_size, 1)
        self.related_score = nn.Linear(self.hidden_size, 1)
        self.turn_token = nn.Linear(self.hidden_size, self.hidden_size)
        self.ref_state_layer = nn.Linear(self.hidden_size, self.hidden_size)
        # add non-infer classifier
        self.ref_fuse = nn.Linear(self.hidden_size * 2, self.n_slot + 1)
        torch.nn.init.xavier_normal_(self.ans_bias)
        torch.nn.init.xavier_normal_(self.ans_vocab)
        self.layernorm = torch.nn.LayerNorm(self.hidden_size)
        self.albert = torch.nn.DataParallel(self.albert, device_ids=[0, 1])             # 编码器
        self.gcn = MultiRelationalGCN(self.hidden_size, layer_nums=2, relation_type=4)  # 多关系卷积网络

        # 图对比学习模块
        self.grace = Model(Encoder, self.hidden_size, self.hidden_size)

    def init_ans_vocab(self, ans_vocab):
        slot_ans_size = ans_vocab.size(1)
        init_vocab = nn.Parameter(torch.FloatTensor(self.args.slot_size, slot_ans_size, self.args.hidden_size),
                                  requires_grad=True)
        self.max_ans_size = ans_vocab.size(-1)
        self.slot_ans_size = ans_vocab.size(1)
        self.eslots = ans_vocab.size(0)
        ans_vocab = ans_vocab.reshape((-1, self.max_ans_size))
        attention_mask = (ans_vocab != 0)
        token_type_ids = torch.zeros_like(ans_vocab)
        ans_vocab_batches = ans_vocab.split(10)
        attention_mask = attention_mask.split(10)
        token_type_ids = token_type_ids.split(10)

        ans_vocab_encoded = []
        for i, ans_batch in enumerate(ans_vocab_batches):
            batch_len = ans_batch.size(0)
            if ans_batch.sum() != 0:
                _, ans_batch_encoded = self.albert(input_ids=ans_batch,
                                                   token_type_ids=token_type_ids[i],
                                                   attention_mask=attention_mask[i])
            else:
                ans_batch_encoded = torch.zeros(batch_len, self.hidden_size)
            ans_vocab_encoded.append(ans_batch_encoded)
        ans_vocab_encoded = torch.cat(ans_vocab_encoded, dim=0)
        ans_vocab_encoded = ans_vocab_encoded.reshape(-1, slot_ans_size, self.hidden_size)
        init_vocab.data.copy_(ans_vocab_encoded)


    # 加权随机算法
    def a_res(self, samples, m):
        """
        :samples: [(item, weight), ...], [(<i, j>, weight)]
        :k: number of selected items
        :returns: [(item, weight), ...]
        """
        heap = []  # [(new_weight, item), ...]
        for sample in samples:
            wi = sample[1]
            ui = random.uniform(0, 1)
            ki = ui ** (1 / wi)
            if len(heap) < m:
                heapq.heappush(heap, (ki, sample))
            elif ki > heap[0][0]:
                heapq.heappush(heap, (ki, sample))
                if len(heap) > m:
                    heapq.heappop(heap)
        return [item[1] for item in heap]

    def weighted_random_dropout_adj(self, x, coo_adj, p): # 单 batch 实现, 即一次只能处理一个图
        '''
        :x: 节点特征矩阵
        :coo_adj: 节点的邻接矩阵
        :p: 想要去掉的边数所占总边数的占比
        '''
        idx = torch.nonzero(coo_adj).T
        # 首先获取节节点的数量
        n = coo_adj.shape[0]
        edges = [] # 存放边的集合
        for i in range(len(idx[0])):
            edges.append([[idx[0][i],idx[1][i]], 0.0])
        # 获得边权重，也就是计算每条边所链接的两个节点的相似度
        for i in range(len(edges)):
            a = edges[i][0][0]
            b = edges[i][0][1]
            edges[i][1] = 1 - spatial.distance.cosine(x[a], x[b])  # 计算边连接的节点的相似度

        m = p * n # 需要删除的边的数量
        re_edges = self.a_res(edges, m) # 获得删除的边
        for e in re_edges:
            a = e[0][0]
            b = e[0][1]
            coo_adj[a][b] = 0
            coo_adj[b][a] = 0
        # 返回删除边后的邻接矩阵
        return coo_adj

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):  # 损失输入为两个特征矩阵
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                          batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)


    def loss(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True, batch_size: int = 0):  # 对比损失
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret


    def forward(self, input_ids, token_type_ids,
                state_positions, attention_mask, slot_mask, first_edge_mask, second_edge_mask, third_edge_mask,
                fourth_edge_mask,
                max_value, op_ids=None, max_update=None, slot_ans_ids=None, position_ids=None, state_ids=None,
                sample_mm=None): # 模型框架流程input_ids:[1,256] atten_mask:[1,256] slot_mask:[1,256] op_ids:[1,30]
        # 编码器 albert
        enc_outputs = self.albert(input_ids=input_ids,
                                  token_type_ids=token_type_ids,
                                  position_ids=position_ids,
                                  attention_mask=attention_mask)  # last_state:[1,256,1024] pool_state:[1, 1024]
        sample_mm = sample_mm.float()  # [1,2,30]
        sequence_output, pooled_output = enc_outputs[:2]  # 序列输出[1,256,1024]，[CLS]输出[1, 1024]
        state_pos = state_positions[:, :, None].expand(-1, -1, sequence_output.size(-1))    # 获取对话状态在序列中的位置
        state_output = torch.gather(sequence_output, 1, state_pos) # 获取对话状态(30 个槽位)的编码特征 [1, 30, 1024]

        # 三种视角
        '''
        first_edge_mask: [1,30,1]
        second_edge_mask: [1,30,30]
        third_edge_mask: [1,30,1]
        fourth_edge_mask: [1,30,30]
        '''
        # 将对话状态输入GCN, 输出 slot_node [1,30,1024], dialogue_node [1,1,1024]
        slot_node, imor_output = self.gcn(state_output, pooled_output, first_edge_mask, second_edge_mask, third_edge_mask, fourth_edge_mask) # # Implicit Mention Oriented Reasoning _:[1,30,1024] imor_output:[1,1,1024]
# ====================新增=============================================================
        # todo: 对比学习, 对 second_edge_mask 与 fourth_edge_mask 做随机边删除
        for i in range(first_edge_mask.shape[0]):
            first_edge_mask[i] = self.weighted_random_dropout_adj(state_output, first_edge_mask[i])
            fourth_edge_mask[i] = self.weighted_random_dropout_adj(start_output, fourth_edge_mask[i]);
        another_slot_node, another_imor_output = self.gcn(state_output, pooled_output, first_edge_mask, second_edge_mask, third_edge_mask, fourth_edge_mask)
        cl_loss = self.loss(slot_node, another_slot_node, True, slot_node.shape[0]) # 对比损失
# ====================================================
        slot_text = self.slot_gate(sequence_output, state_output, slot_mask)    # Slot Name - Dialogue History 槽位-对话历史 [1,1,30,1024]
        history_text = self.history_gate(pooled_output)                         # Current Turn - Dialogue History 当前回合-对话历史 [1024]

        
        


        # 门控融合得到融合表征
        imor_gate = F.sigmoid(self.imor_w(imor_output))  # [1,1,1] [b, b, 1]
        slot_attn = F.sigmoid(self.slot_w(slot_text))    #[1,1,30,1] [b,b,30,1]
        if len(history_text.shape) == 1:                   # 对话历史只有一句，就是刚开始对话
            history_text = history_text.unsqueeze(0).unsqueeze(0) #[1,1,1024]
        history_attn = F.sigmoid(self.history_w(history_text))    # [b,b,1]
        # 计算融合分数
        fuse_score = self.related_score(
            slot_text.mul(slot_attn) + history_text.mul(history_attn).unsqueeze(2).repeat(1, 1, self.n_slot,
                                                                                          1) + imor_output.mul(
                imor_gate).unsqueeze(2)) # [1,1,30,1]

        # 对话选择
        input_ids, attention_mask, history_slot_mask, history_turn_mask, token_type_ids, state_positions, masked_fuse_score, selected_score = self.select_history(
            fuse_score, input_ids, 2, slot_mask, sample_mm) # ? input_ids:[1,2,256] token_type_ids:[1,2,256] selected_score:[1,2,256]
        # 状态生成器部分
        enc_outputs = self.albert(input_ids=input_ids.reshape(-1, input_ids.shape[-1]).long(),
                                  token_type_ids=token_type_ids.reshape(-1, input_ids.shape[-1]).long(),
                                  position_ids=position_ids.repeat(sample_mm.shape[1], 1).long(),
                                  attention_mask=attention_mask.reshape(-1, input_ids.shape[-1]))
        sequence_output, pooled_output = enc_outputs[:2] # ?  sequence_output:[2,256,1024] pooled_output:[2, 1024]

        selected_score = torch.reshape(selected_score, [-1, input_ids.shape[-1]]).unsqueeze(-1) # ? [2, 256, 1]
        sequence_output = selected_score * sequence_output  # ? [2, 256, 1024]

        sequence_output = sample_mm.transpose(-1, -2).bmm( # ? [1,30,256,1024]
            sequence_output.reshape(sample_mm.shape[:2] + tuple([-1]))).reshape(
            (sample_mm.shape[0], self.n_slot, -1, self.hidden_size))
        attention_mask = sample_mm.transpose(-1, -2).bmm(attention_mask) # ? [1,30,256]
        input_ids = sample_mm.transpose(-1, -2).bmm(input_ids)          # ? [1, 30, 256]
        state_pos = state_positions[:, :, :, None].expand(-1, -1, -1, sequence_output.size(-1)).long() # ? [1,30,30,1024]
        state_output = torch.gather(sequence_output, 2, state_pos) # ? [1,30,30,1024] 
        state_output = state_output.masked_select( # ? [1,30,1,1024]
            torch.eye(self.n_slot).cuda().unsqueeze(0).unsqueeze(-1) == 1).reshape(sequence_output.shape[0],
                                                                                   self.n_slot, 1, -1) 
        start_output = self.start_output(sequence_output) # ! [1,30, 256, 1024]
        end_output = self.end_output(sequence_output)       # ! [1, 30, 256, 1024]
        start_atten_m = state_output.mul(start_output).sum(dim=-1).view(sequence_output.shape[0],
                                                                        sequence_output.shape[1], -1) / math.sqrt(
            self.hidden_size) # ! [1, 30, 256]
        end_atten_m = state_output.mul(end_output).sum(dim=-1).view(sequence_output.shape[0], sequence_output.shape[1],
                                                                    -1) / math.sqrt(self.hidden_size) # ! [1, 30, 256]
        start_logits = start_atten_m.masked_fill(attention_mask.squeeze() == 0, -1e9) # ! [1, 30, 256]
        end_logits = end_atten_m.masked_fill(attention_mask.squeeze() == 0, -1e9) # ! [1, 30, 256]
        start_logits_softmax = F.softmax(start_logits[:, :, 1:], dim=-1) # ? [1, 30, 255]
        end_logits_softmax = F.softmax(end_logits[:, :, 1:], dim=-1) # ? [1, 30, 255]
        ques_attn = F.softmax((sequence_output.mul(state_output.view(-1, self.n_slot, 1, self.hidden_size)).sum(
            dim=-1) / math.sqrt(self.hidden_size)).masked_fill(attention_mask.squeeze() == 0, -1e9), dim=-1) # ? [1, 30, 256]
        sequence_pool_output = ques_attn.unsqueeze(-1).mul(sequence_output).sum(dim=-2).squeeze() # ? [30, 1024]
        sequence_pool_output = sequence_pool_output.view(-1, self.args.n_slot, self.hidden_size)  # ? [1, 30, 1024]
        category_ans = sequence_pool_output.transpose(0, 1).bmm(                                  # ? [30, 1, 16]
            self.slot_mm.mm(self.ans_vocab.view(self.eslots, -1)).view(self.n_slot, self.slot_ans_size, -1).transpose(
                -1, -2)) + self.slot_mm.mm(self.ans_bias.squeeze()).unsqueeze(1)
        category_ans = category_ans.transpose(0, 1) # ? [1, 30, 16]
        category_ans = category_ans.masked_fill((self.slot_ans_mask == 1).unsqueeze(0), -1e9) # ? [1, 30, 16]
        category_ans_softmax = F.softmax(category_ans, dim=-1) # ? [1, 30, 16]
        masked_fuse_score = F.softmax(masked_fuse_score.squeeze(-1).transpose(-1, -2), dim=-1) # ? [1, 30, 1]
        return start_logits_softmax, end_logits_softmax, category_ans_softmax, start_logits, end_logits, category_ans, masked_fuse_score, input_ids.long()

    def slot_gate(self, text, slot, slot_mask, proj_layer=None):
        """_summary_

        Args:
            text (_type_): _description_
            slot (_type_): _description_
            slot_mask (_type_): _description_
            proj_layer (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if proj_layer:
            text = proj_layer(text)
        text = text.unsqueeze(0).unsqueeze(2)
        slot = slot.unsqueeze(1).unsqueeze(3)
        slot_mask = (slot_mask.unsqueeze(0).unsqueeze(2).unsqueeze(-1) == 0)
        slotaware_attn = F.softmax(
            text.mul(slot).masked_fill(slot_mask, 0).sum(dim=-1).masked_fill(slot_mask.squeeze(-1), -1e9), dim=-1)
        slotaware_text = text.mul(slotaware_attn.unsqueeze(-1)).sum(dim=-2)
        return slotaware_text

    def history_gate(self, text):
        indice_mat = torch.ones(text.shape[0], text.shape[0])
        mask = (indice_mat - torch.triu(indice_mat, diagonal=1)).cuda()
        text = self.attention(text, mask)
        text_cls_col = text.unsqueeze(0).repeat(text.shape[0], 1, 1)
        text_cls_row = text.unsqueeze(1).repeat(1, text.shape[0], 1)
        cur_text = torch.cat((text_cls_col, text_cls_row), dim=-1)
        history_text = self.related_cls(cur_text).squeeze()
        return history_text

    def attention(self, input, mask=None):
        key = self.attention_key(input)
        query = self.attention_query(input)
        attn = query.mm(key.transpose(-1, -2))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = F.softmax(attn, dim=-1)
        output = attn.mm(input)
        return output

    def gen_transmm(self, input):
        masknum = input[2]
        slot_count = input[0][-1 - masknum:]
        topk = slot_count.shape[0]
        seqlen = slot_count.shape[1]

        history = input[1][- (masknum + 1) * seqlen:]
        histories = []
        token_ids = []

        slot_numel = [sum(si) for si in slot_count]
        token_numel = sum([1 if hi != 0 else 0 for hi in history[seqlen * (topk - 1):]])
        state = history[seqlen * (topk - 1) + slot_numel[-1]: seqlen * (topk - 1) + token_numel].tolist()
        token_id = 0
        for i in range(topk):
            batch_history = history[seqlen * i: seqlen * i + slot_numel[i]]
            token_ids += [token_id] * len(batch_history)
            token_id = 1 - token_id
            histories += batch_history.tolist()
        seqlen = slot_count.shape[1]
        from_index = []
        for i in range(topk):
            from_index += [si + i * seqlen for si in list(range(sum(slot_count[i])))]
        histories = [histories[0]] + histories[1:][-max(0, seqlen - len(state)) + 1:]
        token_ids = token_ids[1:][-max(0, seqlen - len(state)) + 1:]
        token_ids = [token_ids[0]] + token_ids

        state_position = torch.from_numpy(np.array(list(filter(lambda x: state[x - len(histories)] == 30000,
                                                               [i + len(histories) for i in
                                                                range(len(state))])))).unsqueeze(0)
        history_slot_mask = torch.from_numpy(
            np.array([1] * len(histories) + [0] * (max(0, seqlen - len(histories))))).unsqueeze(0)
        histories = histories + state
        token_ids += [token_id] * len(state)
        token_id = 1 - token_id
        if token_ids[0] == 0:
            token_ids = list(map(lambda x: 1 - x, token_ids))
            token_id = 1 - token_id

        history_mask = torch.from_numpy(
            np.array([1] * len(histories) + [0] * (max(0, seqlen - len(histories))))).unsqueeze(0)
        histories += [0] * (max(0, seqlen - len(histories)))
        token_ids += [token_id] * (max(0, seqlen - len(token_ids)))
        token_ids = torch.from_numpy(np.array(token_ids)).unsqueeze(0)
        history = torch.from_numpy(np.array(histories)).unsqueeze(0)
        history_turn_mask = (history == 2).unsqueeze(0)
        return history, history_mask, history_slot_mask, history_turn_mask, token_ids, state_position

    def select_history(self, fuse_score, history, topk, slot_mm, sample_mm):
        """_summary_

        Args:
            fuse_score (_type_): _description_
            history (_type_): _description_
            topk (_type_): _description_
            slot_mm (_type_): _description_
            sample_mm (_type_): _description_

        Returns:
            _type_: _description_
        """
        turn = history.shape[0]
        topk = min(turn, topk)
        seqlen = history.shape[-1]
        indices_mat = torch.ones((fuse_score.shape[0], fuse_score.shape[0])).cuda()
        indices_mat = indices_mat - torch.triu(indices_mat)
        fuse_score = F.sigmoid(fuse_score)
        fuse_score = fuse_score.masked_fill(indices_mat.unsqueeze(-1).unsqueeze(-1) == 0, -1e9)
        masked_fuse_score = fuse_score.transpose(-2, -3)
        max_related_indices = torch.argsort(masked_fuse_score.squeeze(-1), dim=-1, descending=True)[:, :, :(topk - 1)]
        selected_score = torch.gather(masked_fuse_score.squeeze(-1), dim=-1, index=max_related_indices)

        max_related_indices = torch.cat((max_related_indices,
                                         torch.linspace(0, turn - 1, turn).unsqueeze(-1).unsqueeze(-1).repeat(1,
                                                                                                              self.n_slot,
                                                                                                              1).long().cuda()),
                                        dim=-1)
        indices_flatten = max_related_indices.reshape(-1)
        history = history.index_select(dim=0, index=indices_flatten).reshape(
            max_related_indices.shape[:-1] + tuple([history.shape[-1] * topk]))
        slot_mm = slot_mm.index_select(dim=0, index=indices_flatten).reshape(
            max_related_indices.shape + tuple([seqlen]))
        slot_numel = slot_mm.shape[0] * slot_mm.shape[1]
        mm_detach = slot_mm.reshape(tuple([slot_numel]) + slot_mm.shape[2:]).cpu().detach().numpy()
        history = history.reshape(tuple([slot_numel]) + history.shape[2:]).cpu().detach().numpy()
        trans_input = list(zip(mm_detach, history))
        histories, history_masks, history_slot_masks, history_turn_masks, token_ids, state_positions = [], [], [], [], [], []
        for i in range(len(trans_input)):
            mask_num = i // (self.n_slot * topk)
            history, history_mask, history_slot_mask, history_turn_mask, token_id, state_position = self.gen_transmm(
                list(trans_input[i]) + [mask_num])
            histories.append(history)
            history_masks.append(history_mask)
            history_slot_masks.append(history_slot_mask)
            history_turn_masks.append(history_turn_mask)
            token_ids.append(token_id)
            state_positions.append(state_position)
        history_mask = sample_mm.bmm(
            torch.cat(history_masks, dim=0).reshape(max_related_indices.shape[:2] + tuple([seqlen])).cuda().float())
        history = sample_mm.bmm(
            torch.cat(histories, dim=0).reshape(max_related_indices.shape[:2] + tuple([seqlen])).cuda().float())
        history_slot_mask = torch.cat(history_slot_masks, dim=0).reshape(
            max_related_indices.shape[:2] + tuple([seqlen])).cuda().float()
        history_turn_mask = torch.cat(history_turn_masks, dim=0).reshape(
            max_related_indices.shape[:2] + tuple([seqlen])).cuda().float()
        token_ids = sample_mm.bmm(
            torch.cat(token_ids, dim=0).reshape(max_related_indices.shape[:2] + tuple([seqlen])).cuda().float())
        state_position = torch.cat(state_positions, dim=0).reshape(
            max_related_indices.shape[:2] + tuple([-1])).cuda().float()

        selected_score = sample_mm.bmm(selected_score).repeat(1, 1, seqlen)
        if topk == 1:  # 不选历史对话的情况
            selected_score = torch.ones(history.shape, device='cuda').float()

        selected_score = selected_score.masked_fill(selected_score == -1e9, 1)

        return history, history_mask, history_slot_mask, history_turn_mask, token_ids, state_position, fuse_score, selected_score

    def slot_ref(self, trans_history, slot, slot_mm):
        slot = self.ref_state_layer(slot)
        slot_attn = F.softmax(
            trans_history.unsqueeze(1).mul(slot.unsqueeze(2)).sum(dim=-1).masked_fill(slot_mm.unsqueeze(1) == 0, -1e9),
            dim=-1)
        trans_history = trans_history.unsqueeze(1).mul(slot_attn.unsqueeze(-1)).sum(dim=-2)
        fuse_score = self.ref_fuse(torch.cat((slot, trans_history), dim=-1))
        return fuse_score


class MultiRelationalGCN_(nn.Module): # 多层 关系图卷积网络
    def __init__(self, hidden_size, layer_nums, relation_type):
        super(MultiRelationalGCN, self).__init__()
        self.hidden_size = hidden_size
        self.relation_W = nn.ModuleList()
        for i in range(relation_type):
            self.relation_W.append(nn.Linear(self.hidden_size, self.hidden_size))
        self.layer_nums = layer_nums
        self.f_s = nn.Linear(self.hidden_size, self.hidden_size)
        self.f_g = nn.Linear(2 * self.hidden_size, self.hidden_size)

    def forward(self, slot_node, dialogue_node, first_relation_mask, second_relation_mask, third_relation_mask,
                fourth_relation_mask):
        for i in range(self.layer_nums):
            dialogue_node_r = self.f_s(dialogue_node)
            slot_node_r = self.f_s(slot_node)
            multiple_slot_nodes = []
            multiple_dialogue_nodes = []
            for layer in self.relation_W:
                multiple_slot_nodes.append(layer(slot_node))
                multiple_dialogue_nodes.append(layer(dialogue_node))
            # first_edge
            slot_node_r += first_relation_mask.bmm(
                multiple_dialogue_nodes[0].unsqueeze(0).repeat(slot_node_r.shape[0], 1, 1)) / (
                                       first_relation_mask.sum(dim=-1).unsqueeze(-1) + 1e-4)
            dialogue_node_r += (first_relation_mask.transpose(-1, -2).bmm(multiple_slot_nodes[0]) / (
                        first_relation_mask.transpose(-1, -2).sum(dim=-1).unsqueeze(-1) + 1e-4)).sum(dim=0)
            # second_edge
            slot_node_r += second_relation_mask.bmm(multiple_slot_nodes[1]) / (
                        second_relation_mask.sum(dim=-1).unsqueeze(-1) + 1e-4)
            # third_edge
            slot_node_r += third_relation_mask.bmm(
                multiple_dialogue_nodes[2].unsqueeze(0).repeat(slot_node_r.shape[0], 1, 1)) / (
                                       third_relation_mask.sum(dim=-1).unsqueeze(-1) + 1e-4)
            dialogue_node_r += (third_relation_mask.transpose(-1, -2).bmm(multiple_slot_nodes[2]) / (
                        third_relation_mask.transpose(-1, -2).sum(dim=-1).unsqueeze(-1) + 1e-4)).sum(dim=0)
            # fourth edge +=
            slot_node_r += fourth_relation_mask.bmm(multiple_slot_nodes[3]) / (
                        fourth_relation_mask.sum(dim=-1).unsqueeze(-1) + 1e-4)

            slot_update_gate = self.f_g(torch.cat([slot_node_r, slot_node], dim=-1))
            dialogue_forget_gate = self.f_g(torch.cat([dialogue_node_r, dialogue_node], dim=-1))
            slot_node = F.relu(slot_node_r) * slot_update_gate + slot_node * (1 - slot_update_gate)
            dialogue_node = F.relu(dialogue_node_r) * dialogue_forget_gate + dialogue_node * (1 - dialogue_forget_gate)

        return slot_node, dialogue_node
