import torch
from torch import nn


class SumCombine(nn.Module):
    def __init__(self, 
                 n_node,
                 n_input, 
                 n_hop):
        super().__init__()
        self.n_hop = n_hop
        self.Weight = nn.Parameter(torch.empty(n_hop, n_node, 1, n_input))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.Weight)

    def forward(self, input_list):
        hiddens = torch.stack(input_list, dim=2).unsqueeze(0)  # [1, n_node, input_dim, n_input]
        hiddens = hiddens.repeat(self.n_hop, 1, 1, 1)
        weight = torch.softmax(self.Weight, dim=3)
        return (hiddens * weight).sum(dim=3)


class SimpleCombine(nn.Module):
    def __init__(self, 
                 input_dim,
                 n_input, 
                 n_hop, 
                 attn_dim=16):
        super().__init__()
        self.n_input = n_input
        self.n_hop = n_hop
        self.KProj = nn.Parameter(torch.empty(n_hop, input_dim, attn_dim))
        self.QProj = nn.ParameterList()
        for _ in range(n_input):
            self.QProj.append(torch.empty(n_hop, input_dim, attn_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.KProj)
        for weight in self.QProj:
            nn.init.xavier_uniform_(weight)

    def forward(self, input_list):
        ego_attn = torch.matmul(input_list[0], self.KProj).unsqueeze(3)  # [n_hop, n_node, attn_dim, 1]
        attns = []
        for idx, embed in enumerate(input_list):
            attn = torch.matmul(embed, self.QProj[idx]).unsqueeze(3)  # [n_hop, n_node, attn_dim, 1]
            attns.append(attn)
        attns = torch.cat(attns, dim=3)  # [n_hop, n_node, attn_dim, n_input] 
        attns = ego_attn.repeat(1, 1, 1, self.n_input) * attns
        attns = torch.sum(attns, dim=2, keepdim=True)  # [n_hop, n_node, 1, n_input]
        attns = torch.softmax(attns, dim=3)  # [n_hop, n_node, 1, n_input]
        hiddens = torch.stack(input_list, dim=2).unsqueeze(0)  # [1, n_node, input_dim, n_input]
        return torch.sum(attns * hiddens.repeat(self.n_hop, 1, 1, 1), dim=3)  # [n_hop, n_node, output_dim]


class Combine(nn.Module):
    def __init__(self,
                 input_dim_list,
                 output_dim,
                 n_hop,
                 attn_dim=16):
        super().__init__()
        self.n_hop = n_hop
        self.KProj = nn.Parameter(torch.empty(n_hop, input_dim_list[0], attn_dim))
        self.QProj = nn.ParameterList()
        self.VProj = nn.ParameterList()
        for dim in input_dim_list:
            self.QProj.append(torch.ones(n_hop, dim, attn_dim))
            self.VProj.append(torch.ones(n_hop, dim, output_dim))
        # self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.KProj)
        for weight in self.QProj:
            nn.init.xavier_uniform_(weight)
        for weight in self.VProj:
            nn.init.xavier_uniform_(weight)

    def forward(self, input_list):
        ego_attn = torch.matmul(input_list[0], self.KProj).unsqueeze(3)  # [n_hop, n_node, attn_dim, 1]
        attns, hiddens = [], []
        for idx, embed in enumerate(input_list):
            attn = torch.matmul(embed, self.QProj[idx]).unsqueeze(3)  # [n_hop, n_node, attn_dim, 1]
            hidden = torch.matmul(embed, self.VProj[idx]).unsqueeze(3)  # [n_hop, n_node, output_dim, 1]
            attns.append(attn)
            hiddens.append(hidden)
        attns = torch.cat(attns, dim=3)  # [n_hop, n_node, attn_dim, len(input_list)]
        hiddens = torch.cat(hiddens, dim=3)  # [n_hop, n_node, output_dim, len(input_list)]
        attns = ego_attn.repeat(1, 1, 1, len(input_list)) * attns
        attns = torch.sum(attns, dim=2, keepdim=True)# [n_hop, n_node, 1, len(input_list)]
        attns = attns / (attns.max() - attns.min())
        attns = torch.softmax(attns, dim=3)  # [n_hop, n_node, 1, len(input_list)]
        return torch.sum(attns * hiddens, dim=3)  # [n_hop, n_node, output_dim]