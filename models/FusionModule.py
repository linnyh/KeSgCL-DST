import math
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel, BertModel


# 线性融合模块
class LinearFusion(nn.Module):
    def __init__(self, module_input_dim, module_output_dim, dropout=0.1):
        super().__init__()
        self.slot_lookup_fnn = nn.Sequential(nn.Linear(module_input_dim, module_output_dim))

    def forward(self, feature):
        return self.slot_lookup_fnn(feature)


# 交互注意力融合模块
class InteractiveAttentionFusion(nn.Module):
    def __init__(self, in_feature, out_feature, dropout=0.1):
        # in_dim: the dimension fo query vector
        super().__init__()
        self.w = nn.Linear(in_feature, out_feature)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, feature, kg_feature):
        """
        C feature/context [N, L, 2D]
        Q kgfeature          [N, L, 2D]
        """
        feature = self.w(feature)
        alpha_mat = torch.matmul(feature, kg_feature.transpose(-2, -1))
        alpha = F.softmax(alpha_mat, dim=-1)
        Q = self.dropout(alpha)
        out = torch.matmul(Q, kg_feature)  # 获得知识
        return out


if __name__ == "__main__":
    x = torch.randn(30, 7, 300)
    y = torch.randn(30, 7, 300)
    model = InteractiveAttentionFusion()
    out = model(x, y)
    print(out.shape)
    cls = out[:, 0, :]
    print(cls.shape)
