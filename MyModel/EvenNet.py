# derived from https://github.com/Leirunlin/EvenNet/tree/master

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch_geometric.utils import get_laplacian, add_self_loops
from torch_geometric.nn import MessagePassing


class Even_prop(MessagePassing):
    def __init__(self, K, alpha, Init, bias=True, **kwargs):
        super(Even_prop, self).__init__(aggr='add', **kwargs)
        self.K = int(K // 2)
        self.Init = Init
        self.alpha = alpha
        TEMP = alpha * (1 - alpha) ** (2*np.arange(K//2 + 1))
        self.temp = nn.Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        for k in range(self.K+1):
            self.temp.data[k] = self.alpha*(1-self.alpha)**(2*k)

    def forward(self, x, edge_index, edge_weight=None):
        # L=I-D^(-0.5)AD^(-0.5)
        edge_index1, norm1 = get_laplacian(edge_index, edge_weight, normalization='sym', dtype=x.dtype,
                                           num_nodes=x.size(self.node_dim))
        # I-L
        edge_index2, norm2 = add_self_loops(edge_index1, -norm1, fill_value=1., num_nodes=x.size(self.node_dim))

        hidden = x * self.temp[0]
        for k in range(self.K):
            x = self.propagate(edge_index2, x=x, norm=norm2)
            x = self.propagate(edge_index2, x=x, norm=norm2)
            gamma = self.temp[k + 1]
            hidden = hidden + gamma * x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)


class EvenNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_size, out_channels, K, alpha, Init, dprate, dropout):
        super(EvenNet, self).__init__()
        self.lin1 = nn.Linear(in_channels, hidden_size)
        self.lin2 = nn.Linear(hidden_size, out_channels)
        self.prop1 = Even_prop(K, alpha, Init)

        self.Init = Init
        self.dprate = dprate
        self.dropout = dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop1.reset_parameters()
        # self.bn.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)