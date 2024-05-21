import torch
from torch import Tensor
import torch.nn as nn
from torch_geometric.typing import Adj, OptTensor
from torch.nn import Module, Dropout, LayerNorm, Identity
import torch.nn.functional as F
from typing import Tuple
import numpy as np

# from helpers.classes import GumbelArgs, EnvArgs, ActionNetArgs, Pool, DataSetEncoders


# class GumbelArgs(NamedTuple):
#     learn_temp: bool
#     temp_model_type: ModelType
#     tau0: float
#     temp: float
#     gin_mlp_func: Callable


# class EnvArgs(NamedTuple):
#     model_type: ModelType
#     num_layers: int
#     env_dim: int

#     layer_norm: bool
#     skip: bool
#     batch_norm: bool
#     dropout: float
#     act_type: ActivationType
#     dec_num_layers: int
#     pos_enc: PosEncoder
#     dataset_encoders: DataSetEncoders

#     metric_type: MetricType
#     in_dim: int
#     out_dim: int

#     gin_mlp_func: Callable

#     def load_net(self) -> ModuleList:
#         if self.pos_enc is PosEncoder.NONE:
#             enc_list = [self.dataset_encoders.node_encoder(in_dim=self.in_dim, emb_dim=self.env_dim)]
#         else:
#             if self.dataset_encoders is DataSetEncoders.NONE:
#                 enc_list = [self.pos_enc.get(in_dim=self.in_dim, emb_dim=self.env_dim)]
#             else:
#                 enc_list = [Concat2NodeEncoder(enc1_cls=self.dataset_encoders.node_encoder,
#                                                enc2_cls=self.pos_enc.get,
#                                                in_dim=self.in_dim, emb_dim=self.env_dim,
#                                                enc2_dim_pe=self.pos_enc.DIM_PE())]

#         component_list =\
#             self.model_type.get_component_list(in_dim=self.env_dim, hidden_dim=self.env_dim,  out_dim=self.env_dim,
#                                                num_layers=self.num_layers, bias=True, edges_required=True,
#                                                gin_mlp_func=self.gin_mlp_func)

#         if self.dec_num_layers > 1:
#             mlp_list = (self.dec_num_layers - 1) * [Linear(self.env_dim, self.env_dim),
#                                                     Dropout(self.dropout), self.act_type.nn()]
#             mlp_list = mlp_list + [Linear(self.env_dim, self.out_dim)]
#             dec_list = [Sequential(*mlp_list)]
#         else:
#             dec_list = [Linear(self.env_dim, self.out_dim)]

#         return ModuleList(enc_list + component_list + dec_list)


# class ActionNetArgs(NamedTuple):
#     model_type: ModelType
#     num_layers: int
#     hidden_dim: int

#     dropout: float
#     act_type: ActivationType

#     env_dim: int
#     gin_mlp_func: Callable
    
#     def load_net(self) -> ModuleList:
#         net = self.model_type.get_component_list(in_dim=self.env_dim, hidden_dim=self.hidden_dim, out_dim=2,
#                                                  num_layers=self.num_layers, bias=True, edges_required=False,
#                                                  gin_mlp_func=self.gin_mlp_func)
#         return ModuleList(net)



class TempSoftPlus(Module):
    def __init__(self, gumbel_args: GumbelArgs, env_dim: int):
        super(TempSoftPlus, self).__init__()
        model_list =\
            gumbel_args.temp_model_type.get_component_list(in_dim=env_dim, hidden_dim=env_dim, out_dim=1, num_layers=1,
                                                           bias=False, edges_required=False,
                                                           gin_mlp_func=gumbel_args.gin_mlp_func)
        self.linear_model = nn.ModuleList(model_list)
        self.softplus = nn.Softplus(beta=1)
        self.tau0 = gumbel_args.tau0

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor):
        x = self.linear_model[0](x=x, edge_index=edge_index,edge_attr = edge_attr)
        x = self.softplus(x) + self.tau0
        temp = x.pow_(-1)
        return temp.masked_fill_(temp == float('inf'), 0.)


class ActionNet(nn.Module):
    def __init__(self, action_args: ActionNetArgs):
        """
        Create a model which represents the agent's policy.
        """
        super().__init__()
        self.num_layers = action_args.num_layers
        self.net = action_args.load_net()
        self.dropout = nn.Dropout(action_args.dropout)
        self.act = action_args.act_type.get()

    def forward(self, x: Tensor, edge_index: Adj, env_edge_attr: OptTensor, act_edge_attr: OptTensor) -> Tensor:
        edge_attrs = [env_edge_attr] + (self.num_layers - 1) * [act_edge_attr]
        for idx, (edge_attr, layer) in enumerate(zip(edge_attrs[:-1], self.net[:-1])):
            x = layer(x=x, edge_index=edge_index, edge_attr=edge_attr)
            x = self.dropout(x)
            x = self.act(x)
        x = self.net[-1](x=x, edge_index=edge_index, edge_attr=edge_attrs[-1])
        return x


class CoGNN(Module):
    def __init__(self, gumbel_args: GumbelArgs, env_args: EnvArgs, action_args: ActionNetArgs, pool: Pool):
        super(CoGNN, self).__init__()
        self.env_args = env_args
        self.learn_temp = gumbel_args.learn_temp
        if gumbel_args.learn_temp:
            self.temp_model = TempSoftPlus(gumbel_args=gumbel_args, env_dim=env_args.env_dim)
        self.temp = gumbel_args.temp

        self.num_layers = env_args.num_layers
        self.env_net = env_args.load_net()
        self.use_encoders = env_args.dataset_encoders.use_encoders()

        layer_norm_cls = LayerNorm if env_args.layer_norm else Identity
        self.hidden_layer_norm = layer_norm_cls(env_args.env_dim)
        self.skip = env_args.skip
        self.dropout = Dropout(p=env_args.dropout)
        self.drop_ratio = env_args.dropout
        self.act = env_args.act_type.get()
        self.in_act_net = ActionNet(action_args=action_args)
        self.out_act_net = ActionNet(action_args=action_args)

        # Encoder types
        self.dataset_encoder = env_args.dataset_encoders
        self.env_bond_encoder = self.dataset_encoder.edge_encoder(emb_dim=env_args.env_dim, model_type=env_args.model_type)
        self.act_bond_encoder = self.dataset_encoder.edge_encoder(emb_dim=action_args.hidden_dim, model_type=action_args.model_type)

        # Pooling function to generate whole-graph embeddings
        self.pooling = pool.get()

    def forward(self, x: Tensor, edge_index: Adj, pestat, edge_attr: OptTensor = None, batch: OptTensor = None,
                edge_ratio_node_mask: OptTensor = None) -> Tuple[Tensor, Tensor]:
        result = 0

        calc_stats = edge_ratio_node_mask is not None
        if calc_stats:
            edge_ratio_edge_mask = edge_ratio_node_mask[edge_index[0]] & edge_ratio_node_mask[edge_index[1]]
            edge_ratio_list = []

        # bond encode
        if edge_attr is None or self.env_bond_encoder is None:
            env_edge_embedding = None
        else:
            env_edge_embedding = self.env_bond_encoder(edge_attr)
        if edge_attr is None or self.act_bond_encoder is None:
            act_edge_embedding = None
        else:
            act_edge_embedding = self.act_bond_encoder(edge_attr)

        # node encode  
        x = self.env_net[0](x, pestat)  # (N, F) encoder
        if not self.use_encoders:
            x = self.dropout(x)
            x = self.act(x)

        for gnn_idx in range(self.num_layers):
            x = self.hidden_layer_norm(x)

            # action
            in_logits = self.in_act_net(x=x, edge_index=edge_index, env_edge_attr=env_edge_embedding,
                                        act_edge_attr=act_edge_embedding)  # (N, 2)
            out_logits = self.out_act_net(x=x, edge_index=edge_index, env_edge_attr=env_edge_embedding,
                                          act_edge_attr=act_edge_embedding)  # (N, 2)

            temp = self.temp_model(x=x, edge_index=edge_index,
                                   edge_attr=env_edge_embedding) if self.learn_temp else self.temp
            in_probs = F.gumbel_softmax(logits=in_logits, tau=temp, hard=True)
            out_probs = F.gumbel_softmax(logits=out_logits, tau=temp, hard=True)
            edge_weight = self.create_edge_weight(edge_index=edge_index,
                                                  keep_in_prob=in_probs[:, 0], keep_out_prob=out_probs[:, 0])

            # environment
            out = self.env_net[1 + gnn_idx](x=x, edge_index=edge_index, edge_weight=edge_weight,
                                            edge_attr=env_edge_embedding)
            out = self.dropout(out)
            out = self.act(out)

            if calc_stats:
                edge_ratio = edge_weight[edge_ratio_edge_mask].sum() / edge_weight[edge_ratio_edge_mask].shape[0]
                edge_ratio_list.append(edge_ratio.item())

            if self.skip:
                x = x + out
            else:
                x = out

        x = self.hidden_layer_norm(x)
        x = self.pooling(x, batch=batch)
        x = self.env_net[-1](x)  # decoder
        result = result + x

        if calc_stats:
            edge_ratio_tensor = torch.tensor(edge_ratio_list, device=x.device)
        else:
            edge_ratio_tensor = -1 * torch.ones(size=(self.num_layers,), device=x.device)
        return result, edge_ratio_tensor

    def create_edge_weight(self, edge_index: Adj, keep_in_prob: Tensor, keep_out_prob: Tensor) -> Tensor:
        u, v = edge_index
        edge_in_prob = keep_in_prob[v]
        edge_out_prob = keep_out_prob[u]
        return edge_in_prob * edge_out_prob
