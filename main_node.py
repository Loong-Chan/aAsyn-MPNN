"""experiment on node level task"""
import torch
import numpy as np
from torch.optim import Adam
import torch.nn.functional as F
from pytorch_lightning import seed_everything

from MyModel import Simple_backbone, aAsyn_GNN
from MyDataset import NodeDataset
from Utils import accuracy, multihop_neighbors
from args import ARGS


ARGS.device = ARGS.device if torch.cuda.is_available() else 'cpu'

train_val_size = {
    "citeseer": [55, 55],
    "computers": [138, 138],
    "physics": [230, 230],
    "photo": [96, 96],
    "wikics": [117, 117]
    }

if ARGS.dataset in ["cornell", "texas", "wisconsin", "chameleon", "squirrel"]:
    data = NodeDataset(ARGS.dataset, device=ARGS.device, n_trial=ARGS.num_trial)
else:
    data = NodeDataset(ARGS.dataset, device=ARGS.device, n_trial=ARGS.num_trial,
                       type="random", n_train=train_val_size[ARGS.dataset][0],
                       n_val=train_val_size[ARGS.dataset][1], start_seed=42)
khop_neigs = multihop_neighbors(data.edge_index, data.nnode, 5)

BACKBONE = Simple_backbone(input_dim=data.nfeat,
                           n_layer=ARGS.backbone_layer,
                           keep_init=True).to(ARGS.device)

MODEL = aAsyn_GNN(asyn_model=ARGS.asyn_instance,
                  backbone_dims=BACKBONE.output_dims,
                  hidden_dim=ARGS.combine_hidden,
                  output_dim=data.nclass,
                  n_hop=ARGS.num_hop,
                  dropout=ARGS.asyn_dropout).to(ARGS.device)

optimizer = Adam([
    {
        "params": BACKBONE.parameters(),
        "weight_decay": ARGS.backbone_weight_decay
    },
    {
        "params": MODEL.parameters(),
        "weight_decay": ARGS.asyn_weight_decay
    }
    ],
    lr=ARGS.lr)

trail_acc = []
for trial in range(ARGS.num_trial):
    print(f"[Trial {trial + 1}]")
    seed_everything(ARGS.seed)
    train_idx, val_idx, test_idx = data.train_idx[trial], data.val_idx[trial], data.test_idx[trial]
    MODEL.reset_parameters()
    BACKBONE.reset_parameters()

    val_loss, test_acc = [], []
    for epoch in range(ARGS.epoch):
        MODEL.train()
        optimizer.zero_grad()
        output = BACKBONE(data.x, khop_neigs[0])
        output = MODEL(output, khop_neigs)
        loss = F.nll_loss(output[train_idx], data.y[train_idx])
        loss.backward()
        optimizer.step()

        MODEL.eval()
        output = BACKBONE(data.x, khop_neigs[0])
        output = MODEL(output, khop_neigs)
        test_acc.append(accuracy(data.y[test_idx], output[test_idx]).detach().item())
        val_loss.append(F.nll_loss(output[val_idx], data.y[val_idx]).detach().item())
        if ARGS.verbose:
            print(f"[Epoch {epoch:3d}] train loss: {loss:.4f}, test acc: {test_acc[-1]:.4f}")

    acc = test_acc[np.argmin(val_loss)]
    if ARGS.verbose:
        print(f"Test ACC:{acc:.4f}")
    trail_acc.append(acc)

acc_list = np.array(trail_acc)
mean = acc_list.mean()
std = acc_list.std()
print(f"Mean: {mean:.4f}, Std: {std:.4f}", flush=True)
