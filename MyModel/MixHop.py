from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MixHopConv

class MixHop(nn.Module):
    def __init__(self, 
                 input_dim,
                 hidden_dim,
                 output_dim,
                 n_layer,
                 dropout):
        super().__init__()
        self.nlayer = n_layer
        self.dropout = dropout
        input_dims = [input_dim] + [hidden_dim * 3] * (n_layer - 1)
        output_dims = [hidden_dim] * (n_layer - 1) + [output_dim]
        self.convs = nn.ModuleList()
        for ind, outd in zip(input_dims, output_dims):
            self.convs.append(MixHopConv(ind, outd))
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
    
    def forward(self, x, edge_index):
        hidden = self.convs[0](x, edge_index)
        for idx in range(1, self.nlayer):
            hidden = F.relu(hidden)
            hidden = F.dropout(hidden, p=self.dropout, training=self.training)
            hidden = self.convs[idx](hidden, edge_index)
        return F.log_softmax(hidden, dim=-1)
