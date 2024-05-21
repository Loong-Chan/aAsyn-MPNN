import torch.nn as nn
import torch.nn.functional as F
from MyModel.ModelUtils import get_norm_layer

class MLP(nn.Module):
    def __init__(self, 
                 input_dim,
                 hidden_dim,
                 output_dim,
                 n_layer,
                 dropout,
                 norm=None):
        super().__init__()
        self.n_layer = n_layer
        self.dropout = dropout
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.norms.append(get_norm_layer(norm, input_dim=hidden_dim))
        for _ in range(n_layer - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.norms.append(get_norm_layer(norm, input_dim=hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        for norm in self.norms:
            norm.reset_parameters()

    def forward(self, features, *args, **kwargs):
        hidden = self.layers[0](features)
        for idx in range(1, self.n_layer):
            hidden = self.norms[idx - 1](hidden)
            hidden = F.relu(hidden)
            hidden = F.dropout(hidden, p=self.dropout, training=self.training)
            hidden = self.layers[idx](hidden)
        # return F.log_softmax(hidden, dim=-1)
        return hidden
