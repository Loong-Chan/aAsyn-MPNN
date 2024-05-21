import torch
from torch_geometric.nn import global_add_pool, global_mean_pool, BatchNorm

def propagate(features, edge_index, hop=1, mean=True, self_loop=True):

    device = features.device
    n_node, feat_dim = features.shape
    count = torch.ones(features.shape[0], 1).to(device)
    prop_features = torch.cat([features, count], dim=1)

    gather_idx = edge_index.T[:, :1].expand(-1, feat_dim + 1)
    sactter_idx = edge_index.T[:, 1:].expand(-1, feat_dim + 1)

    for idx in range(hop):
        gather_features = prop_features.gather(0, gather_idx)
        ego_features = prop_features if self_loop else torch.zeros([n_node, feat_dim + 1]).to(device)
        prop_features = ego_features.scatter_add(0, sactter_idx, gather_features)

    prop_features, count = torch.split(prop_features, [feat_dim, 1], dim=1)
    count = torch.where(count==0, torch.ones(count.shape).to(device), count)

    if mean:
        prop_features = prop_features / count
        prop_features = torch.where(torch.isnan(prop_features),
                                    torch.zeros_like(prop_features), prop_features)

    return prop_features, count


def get_pooling_layer(pooling):
    if pooling == "sum":
        return global_add_pool
    if pooling == "mean":
        return global_mean_pool
    raise NotImplementedError


def get_norm_layer(norm_type=None, **kwargs):
    if norm_type is None:
        return torch.nn.Identity()
    if norm_type == "batchnorm":
        assert "input_dim" in kwargs
        return BatchNorm(kwargs["input_dim"])
    raise NotImplementedError

def get_activation(active_type, **kwargs):
    if active_type is None:
        return torch.nn.Identity()
    raise NotImplementedError