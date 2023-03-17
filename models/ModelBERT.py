import math
import numpy as np
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)
from copy import deepcopy
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel, BertModel
from models.FusionModule import *
from models.ConceptNet import AttributeHeuristic
from models.GNN import *

# from models.GNN import ATGAT
# from models.GNN import AttentionAdjGAT
# from models.CrossAttention import GateCrossAttention

'''
    定义了模型的各个模块 
'''


class KnowledgeEnb:
    def __init__(self):
        self.attributeHeuristic = AttributeHeuristic('mini.h5')

    def get_Enb(self, sentence):
        emb = self.attributeHeuristic.get_sentence_vector(sentence)
        return torch.tensor(emb)


class UtteranceEncoding(BertPreTrainedModel):  # 语句编码模块, 采用BERT预训练模型, 输入为token id序列, 也被用来编码槽值与槽
    def __init__(self, config):
        super(UtteranceEncoding, self).__init__(config)

        self.config = config
        self.bert = BertModel(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask, token_type_ids, output_attentions=False, output_hidden_states=False):
        return self.bert(input_ids=input_ids,
                         attention_mask=attention_mask,
                         token_type_ids=token_type_ids,
                         output_attentions=output_attentions,
                         output_hidden_states=output_hidden_states)


class SGMultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads  # 每个注意头的维度
        self.h = heads

        # 参数矩阵
        self.q_linear = nn.Linear(d_model, d_model)  # 这里把W的维度都变长了，等价于多组w
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

        self.scores = None

    def max_n_mask(self, scores, n):
        _, indices = torch.sort(scores, dim=-1, descending=True)
        # print(scores.shape) # [batch, head, 30, 30]
        indices = indices[:, :, :, :n]  # [batch, head, 30, 18]
        mask = torch.zeros_like(scores)
        batch_size = scores.shape[0]
        head = scores.shape[1]
        slot = scores.shape[2]
        for b in range(batch_size):
            for h in range(head):
                for i in range(slot):
                    mask[b][h][i][indices[b][h][i]] = 1

        scores = scores.masked_fill(mask == 0, -1e9)  # 关系最强的N个联系会被保留
        return scores

    def attention(self, q, k, v, d_k, mask=None, dropout=None): # mask 为spacy构建的邻接矩阵
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # 注意力的公式

        # TODO 将句法依赖邻接矩阵作为mask矩阵 done
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0.0, -1e9)  # [PAD] 位置置位无穷小
    
        # scores = self.max_n_mask(scores, 18)
        scores = F.softmax(scores, dim=-2)

        if dropout is not None:
            scores = dropout(scores)

        self.scores = scores
        output = torch.matmul(scores, v)  # 输出为注意力矩阵与值矩阵的乘积
        return output

    def forward(self, q, k, v, mask=None): # mask 为spacy构建的饿邻接矩阵
        bs = q.size(0)  # batch size

        # perform linear operation and split into h heads 分头
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output

    def get_scores(self):
        return self.scores


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads  # 每个注意头的维度
        self.h = heads

        # 参数矩阵
        self.q_linear = nn.Linear(d_model, d_model)  # 这里把W的维度都变长了，等价于多组w
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

        self.scores = None

    def attention(self, q, k, v, d_k, mask=None, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # 注意力的公式
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)  # [PAD] 位置置位无穷小
        scores = F.softmax(scores, dim=-2)

        if dropout is not None:
            scores = dropout(scores)

        self.scores = scores  # (batch, 4, 30, 30)
        output = torch.matmul(scores, v)  # 输出为注意力矩阵与值矩阵的乘积
        return output

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)  # batch size

        # perform linear operation and split into h heads 分头
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output

    def get_scores(self):
        return self.scores


class MultiHeadAttentionTanh(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

        self.scores = None

    def attention(self, q, k, v, d_k, mask=None, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        scores = torch.tanh(scores)
        #         scores = torch.sigmoid(scores)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, 0.)
        #         scores = F.softmax(scores, dim=-1)

        if dropout is not None:
            scores = dropout(scores)

        self.scores = scores
        output = torch.matmul(scores, v)
        return output

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output

    def get_scores(self):
        return self.scores


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([deepcopy(module) for _ in range(N)])


class SlotSelfAttention(nn.Module):
    "A stack of N layers"

    def __init__(self, layer, N):  # N层注意力层
        super(SlotSelfAttention, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, mask=None):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        x = self.norm(x)
        return x + self.dropout(sublayer(x))


class SlotAttentionLayer(nn.Module):
    "SlotAttentionLayer is made up of self-attn and feed forward (defined below)"
    """
    每个层包含一个注意力层和一个线性层两个子层，每个子层间都有残差链接
    """

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(SlotAttentionLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)  # 两层残差
        self.size = size
        self.scores = None

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))  # 每个子层有残差链接
        # self.scores = self.self_attn.scores
        return self.sublayer[1](x, self.feed_forward)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.ReLU()  # use gelu or relu

    def forward(self, x):
        return self.w_2(self.dropout(self.gelu(self.w_1(x))))


# Slot-Token Attention
class UtteranceAttention(nn.Module):
    def __init__(self, attn_head, model_output_dim, dropout=0., attn_type="softmax"):
        super(UtteranceAttention, self).__init__()
        self.attn_head = attn_head
        self.model_output_dim = model_output_dim
        self.dropout = dropout
        if attn_type == "tanh":
            self.attn_fun = MultiHeadAttentionTanh(self.attn_head, self.model_output_dim, dropout=0.)
        else:
            self.attn_fun = MultiHeadAttention(self.attn_head, self.model_output_dim, dropout=0.)

    def forward(self, query, value, attention_mask=None):
        num_query = query.size(0)
        batch_size = value.size(0)
        seq_length = value.size(1)

        expanded_query = query.unsqueeze(0).expand(batch_size, *query.shape)
        if attention_mask is not None:
            expanded_attention_mask = attention_mask.view(-1, seq_length, 1).expand(value.size()).float()
            new_value = torch.mul(value, expanded_attention_mask)
            attn_mask = attention_mask.unsqueeze(1).expand(batch_size, num_query, seq_length)
        else:
            new_value = value
            attn_mask = None

        attended_embedding = self.attn_fun(expanded_query, new_value, new_value, mask=attn_mask)

        return attended_embedding


class Decoder(nn.Module):
    def __init__(self, args, model_output_dim, num_labels, slot_value_pos, device):
        super(Decoder, self).__init__()
        self.model_output_dim = model_output_dim
        self.num_slots = len(num_labels)
        self.num_total_labels = sum(num_labels)
        self.num_labels = num_labels
        self.slot_value_pos = slot_value_pos
        self.attn_head = args.attn_head
        self.device = device
        self.args = args
        self.dropout_prob = self.args.dropout_prob
        self.dropout = nn.Dropout(self.dropout_prob)
        self.attn_type = self.args.attn_type
        self.layer_norm = nn.LayerNorm(normalized_shape=model_output_dim, elementwise_affine=False)  # 无学习参数
        self.leak_relu = nn.LeakyReLU(0.1)

        if args.use_gnn:
            self.at_gat_1 = ATGAT(args, n_feat=self.model_output_dim, n_hid=self.model_output_dim,
                                  n_class=self.model_output_dim, n_out=self.model_output_dim, n_heads=args.attn_head,
                                  dropout=args.at_gat_dropout, alpha=args.at_gat_alpha)
            self.at_gat_2 = ATGAT(args, n_feat=self.model_output_dim, n_hid=self.model_output_dim,
                                  n_class=self.model_output_dim, n_out=self.model_output_dim, n_heads=args.attn_head,
                                  dropout=args.at_gat_dropout, alpha=args.at_gat_alpha)
            self.at_gat_3 = ATGAT(args, n_feat=self.model_output_dim, n_hid=self.model_output_dim,
                                  n_class=self.model_output_dim, n_out=self.model_output_dim, n_heads=args.attn_head,
                                  dropout=args.at_gat_dropout, alpha=args.at_gat_alpha)
            self.at_gat_4 = ATGAT(args, n_feat=self.model_output_dim, n_hid=self.model_output_dim,
                                  n_class=self.model_output_dim, n_out=self.model_output_dim, n_heads=args.attn_head,
                                  dropout=args.at_gat_dropout, alpha=args.at_gat_alpha)
            # self.at_gat_5 = ATGAT(args, n_feat=self.model_output_dim, n_hid=self.model_output_dim,
            #                       n_class=self.model_output_dim, n_out=self.model_output_dim, n_heads=args.attn_head,
            #                       dropout=args.at_gat_dropout, alpha=args.at_gat_alpha)
            # self.at_gat_6 = ATGAT(args, n_feat=self.model_output_dim, n_hid=self.model_output_dim,
            #                       n_class=self.model_output_dim, n_out=self.model_output_dim, n_heads=args.attn_head,
            #                       dropout=args.at_gat_dropout, alpha=args.at_gat_alpha)

        # 融合模块
        if args.fusion == "linear":
            self.slot_knowledge_fusion = LinearFusion(self.model_output_dim + 300, self.model_output_dim)
        elif args.fusion == "attn":
            self.slot_knowledge_fusion = InteractiveAttentionFusion()

        # slot utterance attention  槽-语句注意力, 即 slot-Token Attention
        self.slot_utter_attn = UtteranceAttention(self.attn_head, self.model_output_dim, dropout=0.,
                                                  attn_type=self.attn_type)

        # MLP , FNN1
        self.SlotMLP = nn.Sequential(nn.Linear(self.model_output_dim * 2, self.model_output_dim),
                                     nn.ReLU(),
                                     nn.Dropout(p=self.dropout_prob),
                                     nn.Linear(self.model_output_dim, self.model_output_dim))

        self.state_predictor = StatePrediction(args, self.model_output_dim, 30, True)

        # basic modules, attention dropout is 0.1 by default 模型的主要部分
        attn = MultiHeadAttention(self.attn_head, self.model_output_dim)
        ffn = PositionwiseFeedForward(self.model_output_dim, self.model_output_dim,
                                      self.dropout_prob)  # FFN2, 每个注意力层都有一个线性层

        # attention layer, multiple self attention layers 槽自注意力层，计算槽之间的注意力
        self.slot_self_attn = SlotSelfAttention(SlotAttentionLayer(self.model_output_dim, deepcopy(attn),
                                                                   deepcopy(ffn), self.dropout_prob),
                                                self.args.num_self_attention_layer)

        # prediction 预测层
        self.pred = nn.Sequential(nn.Dropout(p=self.dropout_prob),
                                  nn.Linear(self.model_output_dim, self.model_output_dim),
                                  # nn.LeakyReLU(),
                                  nn.LayerNorm(self.model_output_dim))

        # self.gate = nn.Sequential(nn.Dropout(p=self.dropout_prob),
        #                           nn.Linear(self.model_output_dim*2, self.model_output_dim),
        #                           nn.ReLU(),
        #                           nn.Dropout(p=self.dropout_prob),
        #                           nn.Linear(self.model_output_dim, self.model_output_dim),
        #                           nn.LayerNorm(self.model_output_dim))

        # measure 度量方式
        self.distance_metric = args.distance_metric
        if self.distance_metric == "cosine":
            self.metric = torch.nn.CosineSimilarity(dim=-1, eps=1e-08)
        elif self.distance_metric == "euclidean":
            self.metric = torch.nn.PairwiseDistance(p=2.0, eps=1e-06, keepdim=False)

        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.softmax = nn.Softmax(dim=-1)
        self.nll = CrossEntropyLoss(ignore_index=-1)  # 交叉熵损失函数

        self.criterion = nn.BCELoss()
        self.kl_criterion = nn.KLDivLoss(size_average=False)

        self.glue = nn.GELU()
        self.max_pool = nn.MaxPool1d(5, stride=5)
        self.lstm = nn.LSTM(input_size=self.model_output_dim, hidden_size=self.model_output_dim, batch_first=True)
        # self.mlp = nn.Linear(self.model_output_dim*3, self.model_output_dim)
        self.batch_idx = 1
        self.cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-08)

    # 槽值匹配函数，计算最终loss
    def slot_value_matching(self, value_lookup, hidden, target_slots, labels):  # 槽-值匹配
        loss = 0.
        loss_slot = []
        pred_slot = []

        batch_size = hidden.size(0)
        value_emb = value_lookup.weight[0:self.num_total_labels, :]  # 值嵌入矩阵 （1860,768）

        for s, slot_id in enumerate(target_slots):  # note: target_slots are successive 遍历每一个槽
            hidden_label = value_emb[self.slot_value_pos[slot_id][0]:self.slot_value_pos[slot_id][1],
                           :]  # 获得该槽的所有值的嵌入表示
            num_slot_labels = hidden_label.size(0)  # number of value choices for each slot

            _hidden_label = hidden_label.unsqueeze(0).repeat(batch_size, 1, 1).reshape(batch_size * num_slot_labels, -1)
            _hidden = hidden[:, s, :].unsqueeze(1).repeat(1, num_slot_labels, 1).reshape(batch_size * num_slot_labels,
                                                                                         -1)
            # 计算值嵌入与每个槽位的预测输出嵌入的距离
            _dist = self.metric(_hidden_label, _hidden).view(batch_size, num_slot_labels)

            if self.distance_metric == "euclidean":
                _dist = -_dist
            # _dist = _dist + self.cos(_hidden_label, _hidden).view(batch_size, num_slot_labels) #  加了cos

            _, pred = torch.max(_dist, -1)  # 寻找最相似的值
            pred_slot.append(pred.view(batch_size, 1))

            _loss = self.nll(_dist, labels[:, s])

            loss += _loss
            loss_slot.append(_loss.item())

        pred_slot = torch.cat(pred_slot, 1)  # [batch_size, num_slots]

        return loss, loss_slot, pred_slot

    # 状态对比函数
    def update_state_matching(self, pre_state, update_labels):
        # mask = (pre_state > -1).float()
        pre_state = pre_state.view(-1, 30)
        # pre_state_1 = (pre_state > 0.5).float()
        loss_update = nn.M(pre_state.softmax(dim=-1).log(), update_labels.softmax(dim=-1))  # gpu 1
        # loss_update = self.criterion()
        loss_update = loss_update.sum()
        tp = torch.sum((pre_state >= 0.5) & (update_labels == 1))
        tn = torch.sum((pre_state < 0.5) & (update_labels == 0))
        acc_update = (tp + tn)

        return loss_update, acc_update

    # slot_lookup 表示槽嵌入Embedding，value_lookup 表示值嵌入，目前看来都是计算好了的
    def forward(self, sequence_output, cls_header, attention_mask, labels, slot_lookup, value_lookup, update_labels,
                eval_type="train"):

        batch_size = sequence_output.size(0)
        target_slots = list(range(0, self.num_slots))  # slot id 0-29

        # slot utterance attention 将域槽嵌入与序列嵌入做注意力后拼接
        slot_embedding = slot_lookup.weight[target_slots, :]  # select target slots' embeddings  槽嵌入 [30,1068]
        # 融合知识信息
        if self.args.fusion == "linear":
            slot_embedding = self.slot_knowledge_fusion(slot_embedding)
        elif self.args.fusion == "attn":
            slot_embedding = self.slot_knowledge_fusion(slot_embedding[:, :, :768], slot_embedding[:, :, 768:])

        slot_utter_emb = self.slot_utter_attn(slot_embedding, sequence_output,
                                              attention_mask)  # (batch, 30, 768) slot-token attention 的输出

        # concatenate with slot_embedding  拼接slot表示与话语参与的slot表示 (batch, 30, hidden*2=1536)
        slot_utter_embedding = torch.cat((slot_embedding.unsqueeze(0).repeat(batch_size, 1, 1), slot_utter_emb), 2)

        # MLP 将拼接后的slot表示映射回768维 (batch, 30, 768)
        slot_utter_embedding2 = self.SlotMLP(slot_utter_embedding)

        # slot self attention 槽自注意力机制 (batch, 30, 768) 经过自注意力机制的槽表示
        # hidden_slot = self.slot_self_attn(slot_utter_embedding2)
        hidden_slot = slot_utter_embedding2  # 消融

        # 是否需要用图
        if self.args.use_gnn:
            scores = torch.stack([_.self_attn.scores for _ in self.slot_self_attn.layers],
                                 dim=0)  # 得到每一层的注意力矩阵(96,4,30,30)
            scores = scores.view(self.args.num_self_attention_layer, batch_size, self.args.attn_head, 30,
                                 30)  # 得到每一层的注意力矩阵(6,16,4,30,30)
            # torch.save(scores, "/data/lyh/adjs/adj_{}.pt".format(self.batch_idx))
            # self.batch_idx += 1
            scores = torch.sum(scores, dim=0)  # (16, 4, 30, 30) 将同一头的每层的注意力值相加
            scores = torch.softmax(scores, dim=-1)
            # gat_hidden_slot = torch.cat([hidden_slot, slot_utter_embedding2], dim=-1)  # (batch, 30, 768*2)
            # gat + residual
            gat_hidden_slot = slot_utter_embedding2  # hidden_slot  # 20220224 加了残差slot_utt..
            # 残差链接
            # gat_hidden_slot = torch.relu(gat_hidden_slot)  # 新增
            # gat_out = self.at_gat_1(gat_out, scores)
            # gat_out = self.at_gat_2(gat_out, scores) + gat_out
            hidden_slot_gnn1 = self.at_gat_1(gat_hidden_slot,
                                             scores)  # + slot_utter_embedding2 + hidden_slot  #self.at_gat(gat_hidden_slot, scores) + slot_utter_embedding2 + hidden_slot  # (batch, 30 768)
            # hidden_slot = self.layer_norm(hidden_slot)   # 保证分布稳定性
            hidden_slot_gnn2 = self.at_gat_2(hidden_slot_gnn1, scores)  # + hidden_slot_gnn  # 残差
            hidden_slot_gnn3 = self.at_gat_3(hidden_slot_gnn2, scores)
            hidden_slot_gnn4 = self.at_gat_4(hidden_slot_gnn3, scores)
            # hidden_slot_gnn5 = self.at_gat_5(hidden_slot_gnn4, scores)
            # hidden_slot_gnn6 = self.at_gat_6(hidden_slot_gnn5, scores)
            # 池化
            # if self.args.jk_net == "maxpool":

            gnn_state = torch.cat(
                [hidden_slot_gnn1, hidden_slot_gnn2, hidden_slot_gnn3, hidden_slot_gnn4, gat_hidden_slot], dim=-1)
            # gnn_state = torch.cat(
            #     [hidden_slot_gnn4, gat_hidden_slot], dim=-1)

            gnn_state = self.max_pool(gnn_state)

            # lstm 方法
            # elif self.args.jk_net == "lstm":
            #     gnn_state = torch.cat([hidden_slot_gnn1, hidden_slot_gnn2, hidden_slot_gnn3, hidden_slot_gnn4, gat_hidden_slot], dim=-1).view(-1, 30, 5, self.model_output_dim) # (batch, 30, 5, 768)
            #     # gnn_state = gnn_state.transpose(0, 1)  # (30, batch, 5, 768)
            #     lstm_out = []
            #     for i in range(self.args.train_batch_size):
            #         output, (h_t, c_t) = self.lstm(gnn_state[i])  # c_t (30, 768)
            #         lstm_out.append(c_t)
            #     gnn_state = torch.cat(lstm_out, dim=0)  # (batch, 30, 768)
            # 融合
            hidden_slot = torch.cat([hidden_slot, gnn_state], dim=-1)
            hidden_slot = self.glue(hidden_slot)
            # hidden_slot = self.leak_relu(hidden_slot)  # 改用
            # hidden_slot = torch.norm(hidden_slot) # 暂时换成norm

        # 状态预测层
        # pre_state = self.state_predictor(cls_header, hidden_slot)  # hidden_slot (batch, 30)
        # loss_update, acc_update = self.update_state_matching(pre_state, update_labels)
        # prediction 预测层(一个线性层) 输入=输出=(batch,30,768)
        hidden = self.pred(hidden_slot)

        # slot value matching
        # print(value_lookup.weight.shape) (1860,768)
        # print(target_slots) (0,29) list
        # print(labels)
        loss, loss_slot, pred_slot = self.slot_value_matching(value_lookup, hidden, target_slots, labels)
        # loss = loss_update + loss

        return loss, loss_slot, pred_slot  # , acc_update


class AttentionFusion(nn.Module):
    def __init__(self, d_q, d_k, dropout=None):
        super(AttentionFusion, self).__init__()
        self.w1 = nn.Linear(d_q, d_k)
        self.w2 = nn.Linear(d_q + d_k, d_q)
        self.d_k = d_k

    def attention(self, q, k, v, d_k, dropout=None):
        scores = torch.matmul(q.float(), k.transpose(-2, -1).float()) / math.sqrt(d_k)
        score = F.softmax(scores.float(), dim=-1)
        if dropout is not None:
            score = dropout(score)
        self.scores = score
        output = torch.matmul(score.float(), v.float())
        return output

    def forward(self, bert_feature, knowledge_feature):
        bf = self.w1(bert_feature)
        output = self.attention(bf, knowledge_feature, knowledge_feature, self.d_k)
        return self.w2(torch.cat([bert_feature, output], dim=-1))


def PCA_svd(X, k, center=True):
    n = X.size()[0]
    ones = torch.ones(n).view([n, 1])
    h = ((1 / n) * torch.mm(ones, ones.t())) if center else torch.zeros(n * n).view([n, n])
    H = torch.eye(n) - h
    H = H.cuda()
    X = X.cuda()
    X_center = torch.matmul(H.double(), X.double().transpose(-1,-2))
    u, s, v = torch.svd(X_center)
    components = v[:k].t()
    # explained_variance = torch.mul(s[:k], s[:k])/(n-1)
    return components


# 模型架构
class BeliefTracker(nn.Module):
    def __init__(self, args, slot_lookup, value_lookup, num_labels, slot_value_pos, device):
        super(BeliefTracker, self).__init__()

        self.num_slots = len(num_labels)
        self.num_labels = num_labels
        self.slot_value_pos = slot_value_pos
        self.args = args
        self.device = device
        self.slot_lookup = slot_lookup
        self.value_lookup = value_lookup

        # self.slot_lookup_fnn = nn.Linear(1086, 768)

        self.encoder = UtteranceEncoding.from_pretrained(self.args.pretrained_model)  # 编码器，预训练的BERT
        self.model_output_dim = self.encoder.config.hidden_size  # 718
        self.decoder = Decoder(args, self.model_output_dim, self.num_labels, self.slot_value_pos, device)

        self.linear_fusion = nn.Linear(self.model_output_dim + 300, self.model_output_dim)

        # self.attn_fusion = AttentionFusion(self.model_output_dim, 300)
        # self.graph_cross = GraphCross()
        self.gcn1 = GraphConvolution(300, 300)
        self.gcn2 = GraphConvolution(300, 300)
        self.cross_attn = InteractiveAttentionFusion(768, 300)
        self.w_fusion = nn.Linear(1368, 768)

        # if args.cross_attn:
        #     self.cross_attn = GateCrossAttention(768, 1, 0.1)

    def forward(self, input_ids, attention_mask, token_type_ids, labels, part_mask, update, knowledge,
                eval_type="train"):
        # input_ids: [batch_size, number of tokens]

        batch_size = input_ids.size(0)
        num_slots = self.num_slots

        # encoder, a pretrained model, output is a tuple, (batch_size, sequence_length, hidden_size) is the first.
        '''
            last_hidden_state: (batch_size, sequence_length, hidden_size=768) 基于token的表示
            pooler_output: (batch_size, hidden_size) cls标签经过dense 和 tanh 后的输出
            hidden_states: (batch_size, sequence_length, hidden_size)
            attentions: 
        '''
        output = self.encoder(input_ids, attention_mask, token_type_ids)  # bert， token序列的输出
        sequence_output = output[0]
        cls_header = output[1]
        # 融合外部知识
        # sequence_output = torch.cat([sequence_output, knowledge], dim=-1)
        # sequence_output = self.linear_fusion(sequence_output)
        # sequence_output = self.attn_fusion(sequence_output, knowledge)  # 注意力融合?
        gcn_out1 = self.gcn1(F.normalize(knowledge[0], dim=-1), knowledge[1])
        # gcn_out1 = torch.tanh(gcn_out1)
        gcn_out2 = self.gcn2(gcn_out1, knowledge[1])
        # gcn_out2 = torch.tanh(gcn_out2)
        attn_k1 = self.cross_attn(sequence_output, gcn_out1)
        attn_k2 = self.cross_attn(sequence_output, gcn_out2)
        sequence_output = torch.cat([sequence_output, attn_k1, attn_k2], dim=-1)
        sequence_output = self.w_fusion(sequence_output)
        sequence_output = torch.tanh(sequence_output)

        # sequence_output: [batch_size, number of tokens or sequence_length , hidden_size]
        # 拆分bert的输出为3个部分, part_mask 的维度为 [batch_size, number of tokens or sequence_length]

        if self.args.cross_attn:
            sequence_length = part_mask.shape[1]
            attention_mask_history = torch.zeros_like(part_mask)
            attention_mask_state = torch.zeros_like(part_mask)

            attention_mask_history = attention_mask_history.masked_fill(part_mask == 1, 1)
            attention_mask_state = attention_mask_state.masked_fill(part_mask == 2, 1)

            attention_mask = torch.cat([attention_mask_history, attention_mask_state], dim=-1)

            part_mask = part_mask.view(batch_size, -1, 1).expand(batch_size, -1, sequence_output.shape[-1])

            # 1 表示对话历史，2 表示对话状态， 3 表示当前对话
            history_mask = part_mask == 1
            state_mask = part_mask == 2
            dialog_mask = part_mask == 3

            history = sequence_output * history_mask  # [batch_size, number of sequence_length, hidden_size]
            state = sequence_output * state_mask
            dialog = sequence_output * dialog_mask

            sequence_output = self.cross_attn(history, state)
            sequence_output = torch.cat([dialog, sequence_output], dim=1)

        #     loss, loss_slot, pred_slot = self.decoder(sequence_output, None,
        #                                               labels, self.slot_lookup,
        #                                               self.value_lookup, eval_type)
        # # decoder 解码器
        # else:

        loss, loss_slot, pred_slot = self.decoder(sequence_output, cls_header, attention_mask,
                                                  labels, self.slot_lookup,
                                                  self.value_lookup, update, eval_type)

        # calculate accuracy
        accuracy = pred_slot == labels
        acc_slot = torch.true_divide(torch.sum(accuracy, 0).float(),
                                     batch_size).cpu().detach().numpy()  # slot accuracy 槽精度
        acc = torch.sum(
            torch.div(torch.sum(accuracy, 1), num_slots,
                      rounding_mode='floor')).float().item() / batch_size  # joint accuracy 联合精度

        return loss, loss_slot, acc, acc_slot, pred_slot  # , acc_update


class StatePrediction(nn.Module):  # 用语句编码器的cls输出和图模块槽输出预测槽的状态
    def __init__(self, args, hidden_size, slot_num, training):
        super().__init__()
        self.args = args
        self.slot_num = slot_num
        self.training = training
        self.w_g = nn.Linear(hidden_size * 2, hidden_size)
        self.clsf_update = nn.Linear(hidden_size, 1)

    def forward(self, utte_cls, gat_output):
        clss = torch.stack([utte_cls] * self.slot_num, dim=1)
        g_st = torch.sigmoid(self.w_g(torch.cat([clss, gat_output], dim=-1)))  # (batch, 30, 768)
        hidden = clss * g_st + gat_output * (1 - g_st)  # (batch, 30, 768)
        hidden_update = self.clsf_update(
            F.dropout(hidden, p=self.args.mt_dropout, training=self.training))  # (batch, 30, 1)
        prob_update = torch.sigmoid(hidden_update).squeeze()  # (batch, 30)
        # prob_update = torch.sigmoid(self.update_linear(torch.cat(
        #     [hidden_update, torch.cat([torch.zeros_like(hidden_update[:, :1]).cuda(), hidden_update[:, :-1]], 1)],
        #     -1))).squeeze()
        return prob_update  # 得到每一个槽位的更新概率


if __name__ == '__main__':
    utte_cls = torch.randn(2, 20, 768)
    out = PCA_svd(utte_cls, 20)
    print(out.shape)
