from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from MyModel.ModelUtils import get_norm_layer, get_pooling_layer
from MyModel import MLP


class GCN(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 n_layer,
                 dropout,
                 norm):
        super().__init__()
        
        self.n_layer = n_layer
        self.norm = norm
        self.dropout = dropout
        # create conv layers
        self.convs = nn.ModuleList()
        input_dims = [input_dim] + [hidden_dim] * (n_layer - 1)
        output_dims = [hidden_dim] * (n_layer - 1) + [output_dim]
        for in_, out_ in zip(input_dims, output_dims):
            self.convs.append(GCNConv(in_, out_))
        # create norm layers
        self.norms = nn.ModuleList()
        for _ in range(n_layer - 1):
            self.norms.append(get_norm_layer(norm, input_dim=hidden_dim))
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.norm is not None:
            for norm in self.norms:
                norm.reset_parameters()
    
    def forward(self, x, edge_index):
        hidden = self.convs[0](x, edge_index)
        for idx in range(self.n_layer - 1):
            hidden = self.norms[idx](hidden)
            hidden = F.relu(hidden)
            hidden = F.dropout(hidden, p=self.dropout, training=self.training)
            hidden = self.convs[idx + 1](hidden, edge_index)
        return hidden


class GCN_Graph(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 n_layer,
                 dropout,
                 norm,
                 pooling):
        super().__init__()
        self.gnn = GCN(input_dim=input_dim, hidden_dim=hidden_dim, 
                       output_dim=hidden_dim, n_layer=n_layer, 
                       dropout=dropout, norm=norm)
        self.pooling = get_pooling_layer(pooling)
        self.lin = MLP(input_dim=hidden_dim, hidden_dim=hidden_dim, 
                       output_dim=output_dim, n_layer=2,
                       dropout=0., norm=norm)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.gnn.reset_parameters()
        self.lin.reset_parameters()
    
    def forward(self, x, edge_index, batch):
        hidden = self.gnn(x, edge_index)
        hidden = self.pooling(hidden, batch)
        hidden = self.lin(hidden)
        return hidden
