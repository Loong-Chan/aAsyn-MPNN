# Implement for $a$Asyn-MPNN

## Requirement

```
torch 2.0.0
numpy 1.23.5
pytorch-lightning 2.1.2
torch_geometric 2.5.2
```

## Running Code

For Node-level task:

```
python main_node.py --dataset [...] --lr [...] --asyn_weight_decay [...] --asyn_dropout [...] --num_hop [...] --backbone_layer [...]
```

For Graph-level task

```
python main_graph.py --dataset [...] --lr [...] --asyn_weight_decay [...] --pooling [...] --num_hop [...] --backbone_layer [...]
```

## hyper-parameters

### Node-level （backbone=GCN）

|  Dataset  | aAsyn-layer Learning Rate | aAsyn-layer Weight Decay |  Dropout   | Num-hop | Backbone Layer |
| :-------: | :-----------------------: | :----------------------: | :--------: | :-----: | :------------: |
|   Texas   |        6.795*10-3         |        4.044*10-4        | 2.409*10-1 |    2    |       1        |
| Wisconsin |        3.159*10-3         |        5.826*10-3        | 1.007*10-1 |    1    |       1        |
| Squirrel  |        5.616*10-3         |        3.169*10-4        | 7.986*10-1 |    5    |       5        |
| Chameleon |        1.206*10-3         |        2.611*10-6        | 7.380*10-1 |    4    |       8        |
|  Cornell  |        2.476*10-3         |        4.173*10-3        | 3.435*10-1 |    4    |       3        |
|  Wiki CS  |        4.139*10-3         |        5.300*10-4        | 2.257*10-1 |    1    |       2        |
| Citeseer  |        9.637*10-6         |        2.079*10-3        | 2.193*10-1 |    2    |       1        |
| Computer  |        6.525*10-4         |        4.777*10-4        | 7.339*10-1 |    1    |       2        |
|   Photo   |        3.908*10-4         |        9.522*10-5        | 6.621*10-1 |    1    |       3        |
|  Physics  |        6.920*10-4         |        1.958*10-6        | 0.531*10-1 |    2    |       2        |

### Graph-level （backbone=GCN）

| Dataset  | aAsyn-layer Learning Rate | aAsyn-layer Weight Decay | Pooling | Num-hop | Backbone Layer |
| :------: | :-----------------------: | :----------------------: | :-----: | :-----: | :------------: |
|  MUTAG   |        1.022*10-3         |        7.196*10-7        |   sum   |    3    |       4        |
|   NC1    |        2.291*10-3         |        3.547*10-4        |  mean   |    5    |       2        |
|  NC109   |        6.902*10-3         |        7.525*10-6        |   sum   |    5    |       4        |
|    DD    |        1.032*10-3         |        1.569*10-4        |  mean   |    5    |       3        |
| PROTEINS |        2.627*10-3         |        6.207*10-6        |   sum   |    5    |       3        |

