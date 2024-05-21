import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import SimpleConv, GCNConv, TransformerConv
from MyModel.ModelUtils import get_norm_layer


class Simple_backbone(nn.Module):
    def __init__(self, 
                 input_dim,
                 n_layer, 
                 keep_init=True,
                 use_cache=False,
                 norm=None,
                 **kwargs):
        super().__init__()
        self.cache = None
        self.use_cache = use_cache
        self.keep_init = keep_init
        self.output_dims = [input_dim] * (n_layer + self.keep_init)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for idx in range(n_layer):
            self.convs.append(SimpleConv(aggr="mean", combine_root="self_loop"))
            self.norms.append(get_norm_layer(norm, input_dim=self.output_dims[idx]))

        self.reset_parameters()

    def reset_parameters(self):
        self.cache = None
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        if self.use_cache and self.cache is not None:
            embed_list = self.cache
        else:
            embed_list = [x]
            for conv, norm in zip(self.convs, self.norms):
                embed = conv(embed_list[-1], edge_index)
                embed = norm(embed)
                embed_list.append(embed)
            self.cache = embed_list
        return embed_list if self.keep_init else embed_list[1:]


class GCN_backbone(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 n_layer,
                 dropout,
                 keep_init=True,
                 norm=None,
                 **kwargs):
        super().__init__()
        self.n_layer = n_layer
        self.keep_init = keep_init
        self.dropout = dropout
        self.output_dims = [input_dim] * self.keep_init + [output_dim] * n_layer
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for dim in self.output_dims:
            self.convs.append(GCNConv(dim, output_dim))
        for _ in range(n_layer):
            self.norms.append(get_norm_layer(norm, input_dim=output_dim))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        embed_list = [x]
        hidden = self.convs[0](embed_list[-1], edge_index)
        for idx in range(1, self.n_layer):
            hidden = self.norms[idx - 1](hidden)
            hidden = F.relu(hidden)
            embed_list.append(hidden)
            hidden = F.dropout(embed_list[-1], p=self.dropout, training=self.training)
            hidden = self.convs[idx](hidden, edge_index)
        hidden = self.norms[-1](hidden)
        embed_list.append(hidden)
        return embed_list if self.keep_init else embed_list[1:]


# class backbone(nn.Module):
#     raise NotImplementedError


# class GIN_backbone(nn.Module):
#     raise NotImplementedError


# class GT_backbone(nn.Module):
#     raise NotImplementedError
