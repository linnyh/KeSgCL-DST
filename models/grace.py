import torch
import torch.nn as nn
# from layers import GCN, AvgReadout, Discriminator
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)
import torch.nn.functional as F
import numpy as np
import scipy
import scipy.sparse as sp
from scipy.sparse import coo_matrix
import heapq
import random
from scipy import spatial


class GCN(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.use_bias = bias
        if self.use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        if self.use_bias:
            nn.init.zeros_(self.bias)

    def forward(self, input_features, adj):
        support = torch.mm(input_features, self.weight)
        output = torch.spmm(adj, support)  # 一次节点间信息传播
        if self.use_bias:
            return output + self.bias
        else:
            return output


# 定义两层的GCN作为Encoder
class Encoder(torch.nn.Module):
    # in_channels--feature size;out_channels--hid size;activation--relu
    def __init__(self, in_channels: int, out_channels: int, activation,
                 base_model=GCN, k: int = 2):  # 采用 GCN 作为基础模型
        super(Encoder, self).__init__()
        self.base_model = base_model

        assert k >= 2
        self.k = k  # GCN层数
        self.conv = [base_model(in_channels, 2 * out_channels)]
        for _ in range(1, k - 1):
            self.conv.append(base_model(2 * out_channels, 2 * out_channels))
        self.conv.append(base_model(2 * out_channels, out_channels))
        self.conv = nn.ModuleList(self.conv)

        self.activation = activation

    # GNN编码器 输入一个图
    def forward(self, x, adj, sparse=False):
        for i in range(self.k):
            x = self.activation(self.conv[i](x, adj, sparse))
        return x


# 定义模型结构
class Model(torch.nn.Module):
    def __init__(self, encoder: Encoder, num_hidden: int, num_proj_hidden: int,
                 tau: float = 0.5):
        super(Model, self).__init__()
        self.encoder: Encoder = encoder
        self.tau: float = tau

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)
        # The Crowd
        self.lda = 0.5
        self.k = 10  # k =10
        self.hp = 6

    # 输入原始图
    def forward(self, x, adj, sparse):
        # 视图生成
        view_adj = self.dropout_adj(adj, 0.1)        # 随机边删除
        view_feature = self.drop_feature(x, 0.1)   # 随机节点遮蔽

        view_adj = view_adj.to(torch.float32)
        view_feature = view_feature.to(torch.float32)

        return self.encoder(view_feature, adj), self.encoder(x, view_adj)  # 返回两个视图经过GCN编码后的表征

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


    def dropout_adj(self, coo_adj, p): # 接受一个tensor的邻接矩阵
        if p < 0. or p > 1.:
            raise ValueError('Dropout probability has to be between 0 and 1, '
                             'but got {}'.format(p))
        idx = torch.nonzero(coo_adj).T  # 这里需要转置一下
        data = coo_adj[idx[0],idx[1]]

        coo_adj = torch.sparse_coo_tensor(idx, data, coo_adj.shape)
        row = coo_adj._indices()[0]
        col = coo_adj._indices()[1]
        data = coo_adj._values()
        shape = coo_adj.size()
        coo_adj = coo_matrix((data, (row, col)), shape=shape)
        
        coo_adj = coo_adj.tocoo()
        col = list(coo_adj.col.reshape(-1))
        row = list(coo_adj.row.reshape(-1))
        '''np.random.binomial(n,p,size=None)
        n表示n次的试验，p表示的试验成功的概率，n可是是一个float但是也会被变成整数来使用。
        size：可选项，int或者int的元祖，表示的输出的大小，如果提供了size，例如(m,n,k)，那么会返回m*n*k个样本。'''
        obs = np.random.binomial(1, 1 - p, len(col))
        # 使用随机参数们构造一个对角矩阵，用来和原矩阵相乘以mask边
        mask = sp.coo_matrix((obs, (row, col)), shape=(coo_adj.shape[0], coo_adj.shape[0]))
        drop_edge_adj = coo_adj.multiply(mask)
        return torch.from_numpy(drop_edge_adj.toarray())

    def drop_feature(self, x, drop_prob): # 输入一个tensor，维度为 [length, dim]
        # uniform_采样（0，1）数字填充ternsor
        drop_mask = torch.empty(
            (x.size(1),),
            dtype=torch.float32,
            device=x.device).uniform_(0, 1) < drop_prob
        x = x.clone()
        x[:, drop_mask] = 0
        return x

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

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


if __name__ == '__main__':
    model = Model(GCN(20, 20), 20, 20)
    # adj = torch.eye(10,10)
    # coo_adj = coo_matrix.tocoo(adj)
    
    # res = model.dropout_adj(coo_adj, 0.5)
    # print(coo_adj)
    # print(res) 
    adj = torch.eye(10,10)
    print(adj)
    # idx = torch.nonzero(a).T  # 这里需要转置一下
    # data = a[idx[0],idx[1]]

    # coo_a = torch.sparse_coo_tensor(idx, data, a.shape)
    # # res = model.dropout_adj(coo_a, 0.5)
    # row = coo_a._indices()[0]
    # col = coo_a._indices()[1]
    # data = coo_a._values()
    # shape = coo_a.size()
    # adj = coo_matrix((data, (row, col)), shape=shape)
    x = torch.randn(10,20)
    # adj = model.dropout_adj(adj, 0.5)
    x1 = torch.randn(1,20)
    x2 = torch.randn(1,20)
    print(spatial.distance.cosine(x1, x2))
    print(x)
    # x = model.drop_feature(x, 0.5)
    r_adj = model.weighted_random_dropout_adj(x, adj, 0.5)
    print(r_adj)
    output = model(x, r_adj, False)
    print(output)
    print(r_adj)