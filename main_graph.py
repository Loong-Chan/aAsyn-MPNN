"""experiment on node graph task"""
import torch
import numpy as np
from torch.optim import Adam
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from MyModel import aAsyn_GNN_graph, Simple_backbone
from MyDataset import GraphDataset
from Utils import accuracy, multihop_neighbors
from args import ARGS


ARGS.device = ARGS.device if torch.cuda.is_available() else 'cpu'

dataset = GraphDataset(ARGS.dataset)


BACKBONE = Simple_backbone(input_dim=dataset.nfeat,
                            n_layer=ARGS.backbone_layer,
                            use_cache=False).to(ARGS.device)

MODEL = aAsyn_GNN_graph(asyn_model=ARGS.asyn_instance,
                        backbone_dims=BACKBONE.output_dims,
                        hidden_dim=ARGS.combine_hidden,
                        output_dim=dataset.nclass,
                        n_hop=ARGS.num_hop,
                        dropout=ARGS.asyn_dropout,
                        pooling=ARGS.pooling).to(ARGS.device)

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
    seed_everything(42)
    loaders = dataset.random_split_loader(batch_size=ARGS.batch_size,
                                          proportion=ARGS.split)
    train_loader, val_loader, test_loader = loaders
    train_khops, val_khops, test_khops = [], [], []
    for train in train_loader:
        train_khops.append(multihop_neighbors(train.edge_index, train.num_nodes, ARGS.num_hop, ARGS.device))
    for val in val_loader:
        val_khops.append(multihop_neighbors(val.edge_index, val.num_nodes, ARGS.num_hop, ARGS.device))
    for test in test_loader:
        test_khops.append(multihop_neighbors(test.edge_index, test.num_nodes, ARGS.num_hop, ARGS.device))

    scheduler = CosineAnnealingWarmRestarts(optimizer,
                                            T_0=ARGS.scheduler_t_0,
                                            T_mult=ARGS.scheduler_t_multi)

    MODEL.reset_parameters()
    BACKBONE.reset_parameters()

    for epoch in range(ARGS.epoch):
        MODEL.train()
        for idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            batch = batch.to(ARGS.device)
            output = BACKBONE(batch.x, batch.edge_index)
            output = MODEL(output, train_khops[idx], batch.batch)
            loss = F.nll_loss(output, batch.y)
            loss.backward()
            optimizer.step()
        scheduler.step()

    MODEL.eval()
    correct = 0
    for idx, batch in enumerate(test_loader):
        batch = batch.to(ARGS.device)
        output = BACKBONE(batch.x, batch.edge_index)
        output = MODEL(output, test_khops[idx], batch.batch)
        correct = correct + accuracy(batch.y, output, norm=False)
    acc = correct / len(test_loader.dataset)
    if ARGS.verbose:
        print(f"[Epoch {epoch:3d}] test acc: {acc:.4f}")
    trail_acc.append(acc.to("cpu"))

acc_list = np.array(trail_acc)
mean = acc_list.mean()
std = acc_list.std()
print(f"Mean: {mean:.4f}, Std: {std:.4f}", flush=True)