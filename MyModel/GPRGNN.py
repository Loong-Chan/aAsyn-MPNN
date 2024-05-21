# derived from https://github.com/jianhao2016/GPRGNN

import torch
import numpy as np
from torch.nn import Linear, Parameter
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm


class GPR_prop(MessagePassing):
    def __init__(self, K, alpha, **kwargs):
        super(GPR_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.alpha = alpha
        TEMP = alpha*(1-alpha)**np.arange(K+1)
        TEMP[-1] = (1-alpha)**K
        self.temp = Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        for k in range(self.K+1):
            self.temp.data[k] = self.alpha*(1-self.alpha)**k
        self.temp.data[-1] = (1-self.alpha)**self.K

    def forward(self, x, edge_index):
        edge_index, norm = gcn_norm(edge_index, num_nodes=x.size(0), dtype=x.dtype)
        hidden = x * (self.temp[0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k+1]
            hidden = hidden + gamma * x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class GPRGNN(torch.nn.Module):
    def __init__(self, 
                 nfeat, 
                 nhid,
                 nclass,
                 niter=10, 
                 teleport_prop=0.1, 
                 lin_drop=0.5, 
                 prop_drop=0.5):
        super(GPRGNN, self).__init__()
        self.lin1 = Linear(nfeat, nhid)
        self.lin2 = Linear(nhid, nclass)
        self.prop = GPR_prop(niter, teleport_prop)
        self.lin_drop = lin_drop
        self.prop_drop = prop_drop

    def reset_parameters(self):
        self.prop.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.lin_drop, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.lin_drop, training=self.training)
        x = self.lin2(x)
        x = F.dropout(x, p=self.prop_drop, training=self.training)
        x = self.prop(x, edge_index)
        return F.log_softmax(x, dim=1)
