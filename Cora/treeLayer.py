#encoding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from tree_utils import sortNodes

class RecursiverLayer(nn.Module):
    def __init__(self,feat_dim,hidden_dim,num_class,weight_dict,alpha):
        super(RecursiverLayer, self).__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.num_class = num_class
        self.weight_dict = weight_dict
        self.GRU = nn.GRUCell(feat_dim,feat_dim)
        self.a = nn.Parameter(torch.zeros(size=(2 * feat_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.alpha = alpha
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    def forward(self,inputs,adj):
        global neighbor_list,temp_out,x_1,x_2
        temp_outs = {}
        for idx,neighbor_dict in self.weight_dict.items():
            if len(neighbor_dict) == 1:
                neighbor_list = list(neighbor_dict.keys())
                neighbor_list.append(neighbor_list[0])
                node1_idx = neighbor_list[0]
                node2_idx = neighbor_list[1]
                x_1 = inputs[node1_idx,:].reshape(1,-1)
                x_2 = inputs[node2_idx,:].reshape(1,-1)
                temp_out = self.GRU(x_2,x_1)
            elif len(neighbor_dict) == 2:
                neighbor_list = list(neighbor_dict.keys())
                node1_idx = neighbor_list[0]
                node2_idx = neighbor_list[1]
                x_1 = inputs[node1_idx, :].reshape(1, -1)
                x_2 = inputs[node2_idx, :].reshape(1, -1)
                temp_out = self.GRU(x_2, x_1)
            else:
                count = 0
                temp_dict = {}
                neighbor_list = sortNodes(neighbor_dict)
                node1_idx = neighbor_list[0]
                node2_idx = neighbor_list[1]
                x_1 = inputs[node1_idx, :].reshape(1, -1)
                x_2 = inputs[node2_idx, :].reshape(1, -1)
                temp_out = self.GRU(x_2, x_1)
                temp_dict[count] = temp_out
                temp_neighbor_dict = {key:weight for key,weight in neighbor_dict.items() if key != node2_idx and key != node1_idx}
                value = neighbor_dict[node1_idx] + neighbor_dict[node2_idx]
                temp_neighbor_dict[count] = value
                count += 1
                while len(temp_neighbor_dict) < 2:
                    temp_neighbor_list = sortNodes(temp_neighbor_dict)
                    node1_idx = temp_neighbor_list[0]
                    node2_idx = temp_neighbor_list[1]
                    if node1_idx >= 10000:
                        x_1 = temp_dict[node1_idx].reshape(1,-1)
                    else:
                        x_1 = inputs[node1_idx,:].reshape(1,-1)
                    if node2_idx >= 10000:
                        x_2 = temp_dict[node2_idx].reshape(1, -1)
                    else:
                        x_2 = inputs[node2_idx, :].reshape(1, -1)
                    temp_out = self.GRU(x_2,x_1)
                    temp_dict[count] = temp_out
                    value = temp_neighbor_dict[node1_idx] + temp_neighbor_dict[node2_idx]
                    temp_neighbor_dict = {key:weight for key,weight in temp_neighbor_dict.items() if key != node2_idx and key != node1_idx}
                    temp_neighbor_dict[count] = value
                    count += 1
                temp_out = temp_dict[count - 1]
            temp_outs[idx] = temp_out
        outs = torch.zeros_like(inputs)
        for idx, value in temp_outs.items():
            outs[idx, :] = temp_outs[idx]
        N = outs.shape[0]
        a_input = torch.cat([outs.repeat(1, N).view(N * N, -1), outs.repeat(N, 1)], dim=1).view(N, -1, 2 * self.feat_dim)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        return attention