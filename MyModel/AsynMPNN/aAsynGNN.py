from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TransformerConv, GINConv
from MyModel.AsynMPNN.CombineClass import Combine, SimpleCombine
from MyModel.MLP import MLP
from MyModel.ModelUtils import get_pooling_layer, get_norm_layer


class MyGINConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, bn_output=True):
        super().__init__()
        mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.conv = GINConv(mlp, train_eps=True)
        if bn_output:
            self.bn = nn.BatchNorm1d(output_dim)
        else:
            self.bn = nn.Identity()

    def reset_parameters(self):
        self.conv.reset_parameters()
    
    def forward(self, x, edge_index):
        hidden = self.conv(x, edge_index)
        return self.bn(hidden)

        

class aAsynGCNLayer(nn.Module):
    def __init__(self, 
                 input_dim,
                 output_dim,
                 n_hop,
                 dropout,
                 norm=None):
        super().__init__()
        self.n_hop = n_hop
        self.dropout = dropout
        self.norm = norm
        self.ego_lin = nn.Linear(input_dim, output_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.norms.append(get_norm_layer(norm, input_dim=output_dim))
        for _ in range(n_hop):
            self.convs.append(GCNConv(input_dim, output_dim))
            self.norms.append(get_norm_layer(norm, input_dim=output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        self.ego_lin.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        if self.norm is not None:
            for norm in self.norms:
                norm.reset_parameters()

    def forward(self, multi_input, edge_index_list):
        hidden = self.ego_lin(multi_input[0])
        hidden = self.norms[-1](hidden)
        hidden = F.dropout(hidden, self.dropout, self.training)
        for idx in range(self.n_hop):
            prop = self.convs[idx](multi_input[idx + 1], edge_index_list[idx])
            prop = self.norms[idx](prop)
            F.dropout(prop, self.dropout, self.training)
            hidden = hidden + F.relu(prop)
        return hidden


class aAsynGTLayer(nn.Module):
    def __init__(self, 
                 input_dim,
                 output_dim,
                 n_hop,
                 dropout):
        super().__init__()
        self.n_hop = n_hop
        self.dropout = dropout
        self.lin = nn.Linear(input_dim, output_dim)
        self.convs = nn.ModuleList()
        for _ in range(n_hop):
            self.convs.append(TransformerConv(input_dim, output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, multi_input, edge_index_list):
        hidden = self.lin(multi_input[0])
        hidden = F.dropout(hidden, self.dropout, self.training)
        for idx in range(self.n_hop):
            prop = self.convs[idx](multi_input[idx + 1], edge_index_list[idx])
            F.dropout(prop, self.dropout, self.training)
            hidden = hidden + prop
        return hidden


class aAsynGINLayer(nn.Module):
    def __init__(self, 
                 input_dim,
                 output_dim,
                 n_hop,
                 dropout):
        super().__init__()
        self.n_hop = n_hop
        self.dropout = dropout
        self.lin = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )
        self.convs = nn.ModuleList()
        for _ in range(n_hop):
            self.convs.append(MyGINConv(input_dim, output_dim, output_dim, bn_output=False))
        self.reset_parameters()

    def reset_parameters(self):
        self.lin[0].reset_parameters()
        self.lin[3].reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, multi_input, edge_index_list):
        hidden = self.lin(multi_input[0])
        hidden = F.dropout(hidden, self.dropout, self.training)
        for idx in range(self.n_hop):
            prop = self.convs[idx](multi_input[idx + 1], edge_index_list[idx])
            F.dropout(prop, self.dropout, self.training)
            hidden = hidden + prop
        return hidden


class aAsyn_GNN(nn.Module):
    def __init__(self, 
                 backbone_dims,
                 hidden_dim,
                 output_dim,
                 n_hop,
                 asyn_model="GCN",
                 dropout=0.,
                 asyn_norm="batchnorm",
                 asyn_postmap_layer=0,
                 combine="attention"):
        super().__init__()
        self.combine = combine
        self.postmap_layer = asyn_postmap_layer
        # create combine function
        assert combine in ["attention", "simple"]
        if combine == "attention":
            self.combine_layer = Combine(backbone_dims, hidden_dim, n_hop + 1)
        elif combine == "simple":
            self.combine_layer = SimpleCombine(backbone_dims[1], len(backbone_dims)-1, n_hop + 1)
        # create asyn-mpnn layer
        asyn_output_dim = hidden_dim if self.postmap_layer > 0 else output_dim
        if asyn_model == "GCN":
            self.asyn_conv = aAsynGCNLayer(hidden_dim, asyn_output_dim, n_hop, dropout, asyn_norm)
        elif asyn_model == "GT":
            self.asyn_conv = aAsynGTLayer(hidden_dim, asyn_output_dim, n_hop, dropout)
        elif asyn_model == "GIN":
            self.asyn_conv = aAsynGINLayer(hidden_dim, asyn_output_dim, n_hop, dropout)     
        # create postmap
        if self.postmap_layer > 0:
            self.postmap = MLP(hidden_dim, hidden_dim, output_dim, self.postmap_layer, 0., "batchnorm")
        else:
            self.postmap = nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        self.combine_layer.reset_parameters()
        self.asyn_conv.reset_parameters()
        if self.postmap_layer > 0:
            self.postmap.reset_parameters()

    def forward(self, backbone_outputs, khop_edge_index):
        if self.combine == "attention":
            hidden = self.combine_layer(backbone_outputs)
        else:
            hidden = self.combine_layer(backbone_outputs[1:])
        
        hidden = self.asyn_conv(hidden, khop_edge_index)
        # return self.postmap(hidden)
        return F.log_softmax(self.postmap(hidden), dim=1)


class aAsyn_GNN_graph(aAsyn_GNN):
    def __init__(self,
                 backbone_dims,
                 hidden_dim,
                 output_dim,
                 n_hop,
                 asyn_model="GCN",
                 dropout=0.,
                 asyn_norm="batchnorm",
                 asyn_postmap_layer=0,
                 combine="attention",
                 pooling="mean"):
        super().__init__(backbone_dims=backbone_dims, hidden_dim=hidden_dim,
                         output_dim=hidden_dim, n_hop=n_hop, asyn_model=asyn_model,
                         dropout=dropout, asyn_norm=asyn_norm, 
                         asyn_postmap_layer=asyn_postmap_layer, combine=combine)
        self.pooling_layer = get_pooling_layer(pooling)
        self.lin = MLP(hidden_dim, hidden_dim, output_dim, 2, 0., "batchnorm")
        self.reset_parameters()

    def reset_parameter(self):
        super().reset_parameters()
        self.lin.reset_parameters()
    
    def forward(self, backbone_outputs, khop_edge_index, batch):
        hidden = super().forward(backbone_outputs, khop_edge_index)
        hidden = self.pooling_layer(hidden, batch)
        logit = self.lin(hidden)
        return F.log_softmax(logit, dim=1)
