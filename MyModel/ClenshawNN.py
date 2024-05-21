import math
import torch as th
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


def relu(x):
    """This implementation allows subgradient of relu(x) at x = 0 to be 1 instead of 0.
    """
    x[x < -0.0] = 0
    return x

class ChebConv(MessagePassing):
    def __init__(self,
                 in_feats,
                 out_feats,
                 n_layers,
                 lamda=1,
                 weight=True,
                 bias=False
                 ):
        super(ChebConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.lamda = lamda

        self.n_layers =  n_layers
        self.thetas = th.log(lamda / (th.arange(n_layers)+1) + 1)
        _ones = th.ones_like(self.thetas)
        self.thetas = th.where(self.thetas<1, self.thetas, _ones)

        if weight:
            self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
        else:
            self.register_parameter('weight', None)
        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.bias is not None:
            init.zeros_(self.bias)
        if self.weight is not None:
            stdv = 1. / math.sqrt(self._out_feats)
            self.weight.data.uniform_(-stdv, stdv)
    
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def forward(self, edge_index, norm_A, h0, last_h, second_last_h, alpha, l):
        rst = self.propagate(edge_index=edge_index, x=last_h, norm=norm_A)
        rst = alpha * h0 + 2*rst - second_last_h
        theta = self.thetas[l-1]
        rstw = rst
        weight = self.weight
        if weight is not None:
            rstw  = th.matmul(rst, weight)  
        if self.bias is not None:
            rstw  = rstw + self.bias
        rst = theta * rstw + (1 - theta) * rst 
        return rst


class ClenshawNN(nn.Module):
    def __init__(
        self,
        edge_index,
        norm_A,
        in_feats,
        n_hidden,
        n_classes,
        K,
        dropout,
        dropout2,
        lamda,
        dropW=False, 
        dropAct=False
    ):  
        super(ClenshawNN, self).__init__()
        self.edge_index = edge_index
        self.norm_A = norm_A

        self.convs = nn.ModuleList()
        for _ in range(K + 1):
            self.convs.append(
                ChebConv(n_hidden, n_hidden, K, lamda=lamda, weight=not dropW, bias=not dropW)
            )
        self.K = K
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_feats, n_hidden))
        self.fcs.append(nn.Linear(n_hidden, n_classes))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())

        self.dropout = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout2)
        self.init_alphas()

        self.dropAct = dropAct

    def init_alphas(self):
        t = th.zeros(self.K + 1)
        t[0] = 1
        self.alpha_params = nn.Parameter(t.float())

    def forward(self, features, *args, **kwargs):
        x = features

        x = self.dropout(x)
        x = self.fcs[0](x)
        x = relu(x)

        x = self.dropout(x)
        h0 = x
        last_h = th.zeros_like(h0)
        second_last_h = th.zeros_like(h0)
        
        for i, con in enumerate(self.convs):
            '''
            Authors' note: 
                Note that the order of alpha params is INVERSED in clenshaw 
                algorithm. (Check Theorem 3.1)
            '''
            alpha = self.alpha_params[-(i + 1)]
            x = con(self.edge_index, self.norm_A, h0, last_h, second_last_h, alpha, i)
            if not self.dropAct:
                if i < self.K - 1:
                    x = relu(x)
                    x = self.dropout2(x) 
            second_last_h = last_h
            last_h = x

        x = relu(x)
        x = self.dropout(x)
        x = self.fcs[-1](x)

        x = F.log_softmax(x, dim=1)
        return x