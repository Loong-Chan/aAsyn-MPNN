'''default args for running model.'''
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="citeseer")
parser.add_argument("--num_trial", default=10, type=int)
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--epoch", default=300, type=int)
parser.add_argument("--verbose", default=True, type=bool)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--scheduler_t_0", default=5, type=int)
parser.add_argument("--scheduler_t_multi", default=2, type=int)
parser.add_argument("--split", default=[0.8, 0.1, 0.1],
                    help="Only use in graph level task.")
parser.add_argument("--batch_size", default=64,
                    help="Only use in graph level task.")

parser.add_argument("--backbone_layer", default=5, type=int)
parser.add_argument("--backbone_weight_decay", default=0., type=float)
parser.add_argument("--pooling", default="mean", choices=["mean", "sum"],
                    help="Only use in graph level task.")

parser.add_argument("--asyn_instance", default="GCN", choices=["GCN", "GIN", "GT"])
parser.add_argument("--asyn_weight_decay", default=0.01, type=float)
parser.add_argument("--asyn_dropout", default=0., type=float)
parser.add_argument("--num_hop", default=5, type=int, choices=[1, 2, 3, 4, 5])
parser.add_argument("--combine_hidden", default=64, type=int,
                    help="Only Attention Combine Function will use it.")

ARGS = parser.parse_args()
