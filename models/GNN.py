import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ModelBERT import *
import numpy as np
from models.FusionModule import *


# class GraphAttentionLayer(nn.Module):
#     """
#     Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
#     """
#
#     def __init__(self, in_features, out_features, dropout, alpha, concat=True):
#         super(GraphAttentionLayer, self).__init__()
#         self.dropout = dropout
#         self.in_features = in_features
#         self.out_features = out_features
#         self.alpha = alpha
#         self.concat = concat
#
#         self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
#         nn.init.xavier_uniform_(self.W.data, gain=1.414)
#         self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
#         nn.init.xavier_uniform_(self.a.data, gain=1.414)
#
#         self.leakyrelu = nn.LeakyReLU(self.alpha)
#
#     def forward(self, h, adj):  # adj.shape: (N,N)
#         Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
#         e = self._prepare_attentional_mechanism_input(Wh)  # attention matrix
#
#         zero_vec = -9e15 * torch.ones_like(e)
#         attention = torch.where(adj > 0, e, zero_vec)  # 根据邻接矩阵mask得到最终的节点注意力
#         attention = F.softmax(attention, dim=1)
#         attention = F.dropout(attention, self.dropout, training=self.training)
#         h_prime = torch.matmul(attention, Wh)
#
#         if self.concat:
#             return F.elu(h_prime)
#         else:
#             return h_prime
#
#     # 计算注意力矩阵
#     def _prepare_attentional_mechanism_input(self, Wh):
#         # Wh.shape (N, out_feature)
#         # self.a.shape (2 * out_feature, 1)
#         # Wh1&2.shape (N, 1)
#         # e.shape (N, N)
#         Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
#         Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
#         # broadcast add
#         e = Wh1 + Wh2.T
#         return self.leakyrelu(e)
#
#     def __repr__(self):
#         return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    图注意力层
    input: (B,N,C_in)
    output: (B,N,C_out)
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features  # 节点表示向量的输入特征数
        self.out_features = out_features  # 节点表示向量的输出特征数
        self.dropout = dropout  # dropout参数
        self.alpha = alpha  # leakyrelu激活的参数
        self.concat = concat  # 如果为true, 再进行elu激活

        # 定义可训练参数，即论文中的W和a
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # 初始化
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)  # 初始化

        # 定义leakyrelu激活函数
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inp, adj):
        """
        inp: input_fea [B,N, in_features]  in_features表示节点的输入特征向量元素个数
        adj: 图的邻接矩阵  [N, N] 非零即一，数据结构基本知识
        """
        h = torch.matmul(inp, self.W)  # [B, N, out_features]
        N = h.size()[1]  # N 图的节点数

        a_input = torch.cat([h.repeat(1, 1, N).view(-1, N * N, self.out_features), h.repeat(1, N, 1)], dim=-1).view(-1,
                                                                                                                    N,
                                                                                                                    N,
                                                                                                                    2 * self.out_features)
        # [B, N, N, 2*out_features]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))  # (B,N,N)

        # [B, N, N, 1] => [B, N, N] 图注意力的相关系数（未归一化）

        zero_vec = -1e12 * torch.ones_like(e)  # 将没有连接的边置为负无穷

        attention = torch.where(adj > 0, e, zero_vec)  # [B, N, N]
        # 表示如果邻接矩阵元素大于0时，则两个节点有连接，该位置的注意力系数保留，
        # 否则需要mask并置为非常小的值，原因是softmax的时候这个最小值会不考虑。
        attention = F.softmax(attention, dim=1)  # softmax形状保持不变 [B, N, N]，得到归一化的注意力权重！
        attention = F.dropout(attention, self.dropout, training=self.training)  # dropout，防止过拟合
        h_prime = torch.matmul(attention, h)  # [B, N, N].[B, N, out_features] => [B, N, out_features]
        # 得到由周围节点通过注意力权重进行更新的表示
        if self.concat:
            return F.relu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropagation layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class FMLayer(nn.Module):
    def __init__(self, n=10, k=5):
        super(FMLayer, self).__init__()
        self.n = n
        self.k = k
        self.linear = nn.Linear(self.n, 1)  # 前两项线性层
        self.V = nn.Parameter(torch.randn(self.n, self.k))  # 交互矩阵, 为每一个特征初始化一个维度为k的权重参数向量
        nn.init.uniform_(self.V, -0.1, 0.1)

    def fm_layer(self, x):
        linear_part = self.linear(x)
        interaction_part_1 = torch.mm(x, self.V)
        interaction_part_1 = torch.pow(interaction_part_1, 2)
        interaction_part_2 = torch.mm(torch.pow(x, 2), torch.pow(self.V, 2))
        output = linear_part + 0.5 * torch.sum(interaction_part_2 - interaction_part_1, 1, keepdim=False)
        return output

    def forward(self, x):
        return self.fm_layer(x)


class ATGAT(nn.Module):
    def __init__(self, args, n_feat, n_hid, n_class, n_out, dropout, alpha, n_heads):
        """Dense version of GAT
        n_heads 表示有几个GAL层，最后进行拼接在一起，类似self-attention
        从不同的子空间进行抽取特征。
        """
        super(ATGAT, self).__init__()
        self.dropout = dropout
        self.n_heads = n_heads
        self.args = args
        self.batch_idx = 0

        # 定义multi-head的图注意力层
        self.attentions = [GraphAttentionLayer(n_feat, n_hid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)  # 加入pytorch的Module模块
        # 输出层，也通过图注意力层来实现，可实现分类、预测等功能
        self.out_att = GraphAttentionLayer(n_hid * n_heads, n_class, dropout=dropout, alpha=alpha, concat=False)

        self.MLP = nn.Sequential(nn.Linear(n_heads * n_hid, n_out), nn.ReLU())  # 追踪输出层

    def max_n_mask(self, scores, n):
        _, indices = torch.sort(scores, dim=-1, descending=True)
        # print(scores.shape) # [batch, head, 30, 30]
        indices = indices[:, :, :, : self.args.max_n]  # [batch, head, 30, 18]
        mask = torch.zeros_like(scores)
        batch_size = scores.shape[0]
        head = scores.shape[1]
        slot = scores.shape[2]
        for b in range(batch_size):
            for h in range(head):
                for i in range(slot):
                    mask[b][h][i][indices[b][h][i]] = 1
        # scores = scores.masked_fill(mask == 0, 0)  # 关系最强的N个联系会被保留
        # mask = mask.masked_fill(mask >= 3, 1)
        # mask_ = mask.transpose(-1, -2)
        # mask = mask.byte() | mask_.byte()
        return mask

    def max_n_mask_2(self, scores, n):
        _, indices_1 = torch.sort(scores, dim=-1, descending=True)  # 横向关系
        _, indices_2 = torch.sort(scores.transpose(-2, -1), dim=-1, descending=True)  # 纵向关系
        # print(scores.shape) # [batch, head, 30, 30]
        indices_1 = indices_1[:, :, :, : self.args.max_n]  # [batch, head, 30, n]
        indices_2 = indices_2[:, :, :, : self.args.max_n]  # [batch, head, 30, n]

        mask1 = torch.zeros_like(scores)  # 横向
        mask2 = torch.zeros_like(scores)  # 纵向

        batch_size = scores.shape[0]
        head = scores.shape[1]
        slot = scores.shape[2]
        for b in range(batch_size):
            for h in range(head):
                for i in range(slot):
                    mask1[b][h][i][indices_1[b][h][i]] = 1
                    mask2[b][h][i][indices_2[b][h][i]] = 1

        mask2 = mask2.transpose(-2, -1)
        mask = mask1.byte() | mask2.byte()
        return mask.float()

    def forward(self, x, adj):  # （batch,30, 768) , （batch, n_head, 30 ,30 )
        # torch.save(adj, "/data/lyh/adjs/adj_{}.pt".format(self.batch_idx))
        adj = torch.transpose(self.max_n_mask(adj, self.args.max_n), 0, 1)  # (n_head, batch, 30, 30)
        # print(adj)
        # torch.save(adj, "/data/lyh/adjs/adj_{}.pt".format(self.batch_idx))

        self.batch_idx += 1
        # np.savez_compressed("/test/enc_{}".format(batch_idx), codecs)  # 也可以用进行压缩存储

        # x = F.dropout(x, self.dropout, training=self.training)  # dropout，防止过拟合
        x = torch.cat([att(x, adj[i]) for i, att in enumerate(self.attentions)], dim=2)  # 将每个head得到的表示进行拼接
        # x = F.dropout(x, self.dropout, training=self.training)  # dropout，防止过拟合

        x = F.elu(self.out_att(x, adj))  # (nhead, batch, 30, 768)
        x = torch.cat([x[i] for i in range(self.n_heads)], dim=-1)  # (batch, 30, 768 * n_head)
        return self.MLP(x)


class AttentionAdjGAT(nn.Module):
    def __init__(self, nhead, nfeat, nhid, n):
        self.nhead = nhead
        self.nhid = nhid
        self.n = n
        self.n_head_gat = [ATGAT(nfeat, nhid, n_class=None, dropout=0.1, alpha=0.1, n_heads=nhead) for _ in
                           range(nhead)]
        for i, gat in enumerate(self.n_head_gat):
            self.add_module('gat_head{}'.format(i), gat)

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
        # scores = scores.masked_fill(mask == 0, 0)  # 关系最强的N个联系会被保留
        mask = mask.sum(0)
        mask = mask.masked_fill(mask >= 3, 1)
        mask = mask.squeeze(0)
        return mask

    def forward(self, x, scores):
        adj = self.max_n_mask(scores, self.n)  # (16, 4, 30, 30)
        x = self.n_head_gat(x, adj)  # [B, N, out_features]
        return x


class Gate(nn.Module):  # 用语句编码器的cls输出和图模块槽输出预测槽的状态
    def __init__(self, hidden_size, slot_num, training):
        super().__init__()

        self.slot_num = slot_num
        self.training = training
        self.w_g = nn.Linear(hidden_size * 2, hidden_size)
        self.clsf_update = nn.Linear(hidden_size, 1)

    def forward(self, utte_cls, gat_output):
        # utte_cls = utte_cls.unsqueeze(1)
        clss = torch.stack([utte_cls] * 30, dim=1)
        g_st = torch.sigmoid(self.w_g(torch.cat([clss, gat_output], dim=-1)))  # (batch, 30, 768)
        hidden = clss * g_st + gat_output * (1 - g_st)  # (batch, 30, 768)
        hidden_update = torch.tanh(self.clsf_update(F.dropout(hidden, p=0.1, training=self.training)))  # (batch, 30, 1)
        prob_update = torch.sigmoid(hidden_update).squeeze()  # (batch, 30)
        # prob_update = torch.sigmoid(self.update_linear(torch.cat(
        #     [hidden_update, torch.cat([torch.zeros_like(hidden_update[:, :1]).cuda(), hidden_update[:, :-1]], 1)],
        #     -1))).squeeze()
        return prob_update  # 得到每一个槽位的更新概率


class GraphConvolution_(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom

        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.kaiming_uniform_(self.weight, a=0, mode='fan_in', nonlinearity='relu')
        self.activate = nn.Tanh()
        self.drop = nn.Dropout(p=0.2)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        # text = F.normalize(text, dim=-1)
        hidden = torch.matmul(text, self.weight)
        hidden = self.drop(hidden)
        hidden = self.activate(hidden)
        # print('adj:'+str(adj.size()))
        # denom = torch.sum(adj, dim=-1, keepdim=True) + 1
        output = torch.matmul(adj, hidden)
        hidden = self.drop(hidden)
        # output = F.normalize(output, p=2, dim=-1)
        output = self.activate(output)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GraphCross(nn.Module):
    def __init__(self):
        super(GraphCross, self).__init__()
        self.b_gcn1 = GraphConvolution(768, 768)
        self.w1 = nn.Parameter(torch.Tensor(1068, 768))
        self.w2 = nn.Parameter(torch.Tensor(1068, 768))
        torch.nn.init.kaiming_uniform_(self.w1, a=0, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform_(self.w2, a=0, mode='fan_in', nonlinearity='leaky_relu')

        self.b_gcn2 = GraphConvolution(768, 768)
        self.k_gcn1 = GraphConvolution(300, 300)
        self.k_gcn2 = GraphConvolution(300, 300)
        self.fusion = InteractiveAttentionFusion(768, 300)

    def forward(self, b_feature, b_adj, k_feature, k_adj):
        b_1 = self.b_gcn1(b_feature, b_adj)  # 768
        k_1 = self.k_gcn1(k_feature, k_adj)  # 300
        b_1 = torch.cat([b_1, self.fusion(b_1, k_1)], dim=-1)  # 768 + 300
        b_1 = F.normalize(b_1)
        b_1 = torch.matmul(b_1, self.w1)  # 第一次交互融合 768
        b_2 = self.b_gcn2(b_1, b_adj)
        k_2 = self.k_gcn2(k_1, k_adj)
        b_2 = torch.cat([b_2, self.fusion(b_2, k_2)], dim=-1)
        b_2 = torch.matmul(b_2, self.w2)  # 第二次交互融合 768
        return b_2


class SlotRelationGCN(nn.Module):
    def __init__(self):
        super(SlotSelfAttention, self).__init__()


if __name__ == "__main__":
    # torch.set_printoptions(profile="full")
    gc = GraphCross()
    b = torch.rand(2, 10, 768)
    k = torch.rand(2, 15, 300)

    b_adj = torch.rand(2, 10, 10)
    k_adj = torch.rand(2, 15, 15)

    zero_b = torch.zeros_like(b_adj)
    one_b = torch.ones_like(b_adj)
    b_adj = torch.where(b_adj >= 0.5, one_b, zero_b)

    zero_k = torch.zeros_like(k_adj)
    one_k = torch.ones_like(k_adj)
    k_adj = torch.where(k_adj >= 0.5, one_k, zero_k)


    print(b)
    print(k)
    print(b_adj)
    print(k_adj)
    out = gc(b, b_adj, k, k_adj)
    # fm = FMLayer(1068, 6)
    # out = fm(input)
    print(out)
    print(out.shape)

    # adj = torch.transpose(adj, 0, 1)
