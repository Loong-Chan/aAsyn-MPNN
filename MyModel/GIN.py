import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d as BatchNorm
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.nn import GINConv


class GIN(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayer, dropout):
        super().__init__()
        self.nlayer = nlayer
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for i in range(nlayer):
            mlp = Sequential(
                Linear(nfeat, 2 * nhid),
                BatchNorm(2 * nhid),
                ReLU(),
                Linear(2 * nhid, nhid),
            )
            conv = GINConv(mlp, train_eps=True)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(nhid))
            nfeat = nhid

        self.lin1 = Linear(nhid, nhid)
        self.batch_norm1 = BatchNorm(nhid)
        self.lin2 = Linear(nhid, nclass)


    def forward(self, x, edge_index):
        hidden = self.convs[0](x, edge_index)
        hidden = self.batch_norms[0](hidden)
        for idx in range(1, self.nlayer):
            hidden = F.relu(hidden)
            hidden = self.convs[idx](hidden, edge_index)
            if idx < self.nlayer - 1:
                hidden = self.batch_norms[idx](hidden)
        return F.log_softmax(hidden, dim=-1)
        # for conv, batch_norm in zip(self.convs, self.batch_norms):
        #     x = F.relu(batch_norm(conv(x, edge_index)))        
        # x = global_add_pool(x, batch)
        # x = F.relu(self.batch_norm1(self.lin1(x)))
        # x = F.dropout(x, p=0.5, training=self.training)
        # x = self.lin2(x)

