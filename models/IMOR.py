import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiRelationalGCN(nn.Module):
    def __init__(self, hidden_size, layer_nums, relation_type):
        super(MultiRelationalGCN, self).__init__()

        self.hidden_size = hidden_size
        self.f_rs = nn.ModuleList()
        for i in range(relation_type): # 每一种边关系对应一个线性层
            self.f_rs.append(nn.Linear(self.hidden_size, self.hidden_size))
        self.layer_nums = layer_nums
        self.f_s = nn.Linear(self.hidden_size, self.hidden_size)
        self.f_g = nn.Linear(2 * self.hidden_size, self.hidden_size)

    def forward(self, slot_node: torch.Tensor, dialogue_node: torch.Tensor, update_current_mm: torch.Tensor,
                slot_all_connect: torch.Tensor, update_mm: torch.Tensor, slot_domain_connect: torch.Tensor):
        """_summary_
        Args:
            slot_node (torch.Tensor): 槽位节点 [batch_size, 30, 1024]
            dialogue_node (torch.Tensor): 对话节点 [batch_size, 1024], 原始输入为 [CLS] 位表征
            update_current_mm (torch.Tensor): 要更新的插槽和当前回合对话之间的连接;
            slot_all_connect (torch.Tensor): 属于同一域的槽位之间建立连接；
            update_mm (torch.Tensor): 要更新的插槽和其他插槽之间建立连接；
            slot_domain_connect (torch.Tensor): 该边是为了在属于同一域的槽位之间建立连接；
        Returns:
            _type_: _description_
        """
        dialogue_node = dialogue_node.unsqueeze(0).repeat(dialogue_node.shape[0], 1, 1) # [1, 1, 1024]
        for i in range(self.layer_nums):
            dialogue_node_current = self.f_s(dialogue_node)# [1, 1, 1024]
            slot_node_current = self.f_s(slot_node) # [1, 30, 1024]

            relation_dialogue_node_neighbour = []
            relation_slot_node_neighbour = []

            for f_r in self.f_rs:
                relation_dialogue_node_neighbour.append(f_r(dialogue_node)) # ! 4 x [1, 1, 1024]
                relation_slot_node_neighbour.append(f_r(slot_node)) # ! 4 x [1, 30, 1024]

            update_current_mm_d2s = update_current_mm.matmul(relation_dialogue_node_neighbour[0]) / (                       # ! [1, 30, 1024]
                        update_current_mm.sum(-1, keepdim=True).expand_as(slot_node_current) + 1e-4)
            update_current_mm_s2d = update_current_mm.transpose(1, 2).matmul(relation_slot_node_neighbour[0]) / (           # ! [1, 30, 1024]
                        update_current_mm.transpose(1, 2).sum(-1, keepdim=True).expand_as(dialogue_node_current) + 1e-4)

            slot_all_connect_s2s = slot_all_connect.matmul(relation_slot_node_neighbour[1]) / (                             # ! [1, 30, 1024]
                        slot_all_connect.sum(-1, keepdim=True).expand_as(slot_node_current) + 1e-4)

            update_mm_d2s = update_mm.matmul(relation_dialogue_node_neighbour[2]) / (                                       # ! [1, 30, 1024]
                        update_mm.sum(-1, keepdim=True).expand_as(slot_node_current) + 1e-4)
            update_mm_s2d = update_mm.transpose(1, 2).matmul(relation_slot_node_neighbour[2]) / (                           # ! [1, 1, 1024]
                        update_mm.transpose(1, 2).sum(-1, keepdim=True).expand_as(dialogue_node_current) + 1e-4)

            slot_domain_connect_s2s = slot_domain_connect.matmul(relation_slot_node_neighbour[3]) / (                       # ! [1, 30, 1024]
                        slot_domain_connect.sum(-1, keepdim=True).expand_as(slot_node_current) + 1e-4)

            dialogue_node_current = dialogue_node_current + update_current_mm_s2d + update_mm_s2d # ! [1, 1, 1024]
            slot_node_current = slot_node_current + update_current_mm_d2s + slot_all_connect_s2s + update_mm_d2s + slot_domain_connect_s2s # ! [1, 30, 1024]

            # 门控更新
            slot_gate = F.sigmoid(self.f_g(torch.cat([slot_node_current, slot_node], dim=-1))) # ! [1, 30, 1024]
            slot_node = (F.relu(slot_node_current, inplace=False) * slot_gate) + (slot_node * (1 - slot_gate)) # ! [1, 30, 1024]

            dialogue_gate = F.sigmoid(self.f_g(torch.cat([dialogue_node_current, dialogue_node], dim=-1))) # ! [1, 1, 1024]
            dialogue_node = (F.relu(dialogue_node_current, inplace=False) * dialogue_gate) + (dialogue_node * (1 - dialogue_gate)) # ! [1, 1, 1024]

        return slot_node, dialogue_node # ?[1, 30, 1024] [1, 1, 1024]
