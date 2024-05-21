# derived from https://github.com/ytchx1999/PyG-JK-Nets


import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import JumpingKnowledge


class JKNet(nn.Module):
    def __init__(self, nfeat, nhid, nclass, mode='max', nlayer=6, dropout=0.5):
        super(JKNet, self).__init__()
        self.nlayer = nlayer
        self.mode = mode
        self.conv0 = GCNConv(nfeat, nhid)
        self.dropout0 = nn.Dropout(p=dropout)
        for i in range(1, self.nlayer):
            setattr(self, 'conv{}'.format(i), GCNConv(nhid, nhid))
            setattr(self, 'dropout{}'.format(i), nn.Dropout(p=0.5))

        self.jk = JumpingKnowledge(mode=mode)
        if mode == 'max':
            self.fc = nn.Linear(nhid, nclass)
        elif mode == 'cat':
            self.fc = nn.Linear(nlayer * nhid, nclass)

    def forward(self, x, edge_index):
        layer_out = []
        for i in range(self.nlayer):
            conv = getattr(self, 'conv{}'.format(i))
            dropout = getattr(self, 'dropout{}'.format(i))
            x = dropout(F.relu(conv(x, edge_index)))
            layer_out.append(x)
        h = self.jk(layer_out)
        h = self.fc(h)
        h = F.log_softmax(h, dim=1)
        return h
