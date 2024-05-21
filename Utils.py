import torch
from torch_sparse import SparseTensor
from torch_geometric.utils import spmm
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def accuracy(labels, output, norm=True):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels) if norm else correct


def multilabel_cross_entropy(pred, true):
    bce_loss = torch.nn.BCEWithLogitsLoss()
    is_labeled = true == true  # Filter our nans.
    return bce_loss(pred[is_labeled], true[is_labeled].float())

def eval_ap(y_true, y_pred):
    ap_list = []
    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if (y_true[:, i] == 1).sum() > 0 and (y_true[:, i] == 0).sum() > 0:
            # ignore nan values
            is_labeled = y_true[:, i] == y_true[:, i]
            ap = average_precision_score(y_true[is_labeled, i],
                                         y_pred[is_labeled, i])
            ap_list.append(ap)
    if len(ap_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute Average Precision.')
    return sum(ap_list) / len(ap_list)


def add_selfloop(edge_index, n_node):
    selfloop = torch.arange(n_node).repeat(2, 1)
    sl_edge_index = torch.cat([edge_index, selfloop], dim=1)
    return torch.unique(sl_edge_index, dim=1)


def remove_selfloop(edge_index):
    select_idx = (edge_index[0] - edge_index[1]).nonzero().reshape(-1)
    return torch.index_select(edge_index, 1, select_idx)


def multihop_neighbors(edge_index, num_nodes, max_hop=0, device=None):
    device = edge_index.device if device is None else device
    khop_neigs = [edge_index.to(device)]
    sp_adj = SparseTensor(row=edge_index[0], col=edge_index[1],
                          sparse_sizes=(num_nodes, num_nodes)).to("cpu")
    curr_edges = sp_adj.to_dense() + torch.eye(num_nodes)
    for _ in range(max_hop - 1):
        next_edges = spmm(sp_adj, curr_edges)
        edges = next_edges.to(torch.bool) ^ curr_edges.to(torch.bool)
        khop_neigs.append(edges.nonzero().T.to(device))
        curr_edges = next_edges
    return khop_neigs


def propagate(features, edge_index, hop=1, mean=True, self_loop=True):

    n_node, feat_dim = features.shape
    count = torch.ones(features.shape[0], 1)
    prop_features = torch.cat([features, count], dim=1)

    gather_idx = edge_index.T[:, :1].expand(-1, feat_dim + 1)
    sactter_idx = edge_index.T[:, 1:].expand(-1, feat_dim + 1)

    for idx in range(hop):
        gather_features = prop_features.gather(0, gather_idx)
        ego_features = prop_features if self_loop else torch.zeros([n_node, feat_dim + 1])
        prop_features = ego_features.scatter_add(0, sactter_idx, gather_features)

    prop_features, count = torch.split(prop_features, [feat_dim, 1], dim=1)
    count = torch.where(count==0, torch.ones(count.shape), count)

    if mean:
        prop_features = prop_features / count
        prop_features = torch.where(torch.isnan(prop_features),
                                    torch.zeros_like(prop_features), prop_features)

    return prop_features, count


def batch_propagate(features, edge_index, hop=1, mean=True, self_loop=True):

    n_node, feat_dim = features.shape
    count = torch.ones(features.shape[0], 1)
    prop_features = torch.cat([features, count], dim=1)

    gather_idx = edge_index.T[:, :1].expand(-1, feat_dim + 1)
    sactter_idx = edge_index.T[:, 1:].expand(-1, feat_dim + 1)

    for idx in range(hop):
        gather_features = prop_features.gather(0, gather_idx)
        ego_features = prop_features if self_loop else torch.zeros([n_node, feat_dim + 1])
        prop_features = ego_features.scatter_add(0, sactter_idx, gather_features)

    prop_features, count = torch.split(prop_features, [feat_dim, 1], dim=1)
    count = torch.where(count==0, torch.ones(count.shape), count)

    if mean:
        prop_features = prop_features / count
        prop_features = torch.where(torch.isnan(prop_features),
                                    torch.zeros_like(prop_features), prop_features)

    return prop_features, count

