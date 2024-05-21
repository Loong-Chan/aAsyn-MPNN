import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TransformerConv

class MPNN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayer, dropout, **kwargs):
        super().__init__()
        
        self.nfeat = nfeat
        self.nhid = nhid
        self.nclass = nclass
        self.nlayer = nlayer
        self.dropout = dropout

        self.preInitialize()

        self.convs = nn.ModuleList()
        self.convs.append(self.convLayer(nfeat, nhid, idx=0))
        for idx in range(1, nlayer - 1):
            self.convs.append(self.convLayer(nhid, nhid, idx=idx))
        self.convs.append(self.convLayer(nhid, nclass, idx=nlayer - 1))

    def convLayer(self, *args, **kwargs):
        raise NotImplementedError()

    def preInitialize(self, *args, **kwargs):
        raise NotImplementedError()

    def forward(self, features, edge_index):
        hidden = F.dropout(features, p=self.dropout, training=self.training)
        hidden = self.convs[0](hidden, edge_index)
        for idx in range(1, self.nlayer):
            hidden = F.relu(hidden)
            hidden = F.dropout(hidden, p=self.dropout, training=self.training)
            hidden = self.convs[idx](hidden, edge_index)
        return F.log_softmax(hidden, dim=-1)


class GCN(MPNN):
    def __init__(self, nfeat, nhid, nclass, nlayer, dropout):
        super().__init__(nfeat, nhid, nclass, nlayer, dropout)
    
    def convLayer(self, in_channels, out_channels, **kwargs):
        return GCNConv(in_channels, out_channels)

    def preInitialize(self, *args, **kwargs):
        pass


class GT(MPNN):
    def __init__(self, nfeat, nhid, nclass, nlayer, dropout):
        super().__init__(nfeat, nhid, nclass, nlayer, dropout)
    
    def convLayer(self, in_channels, out_channels, **kwargs):
        return TransformerConv(in_channels, out_channels)

    def preInitialize(self, *args, **kwargs):
        pass