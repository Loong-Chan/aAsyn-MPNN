import torch
from torch_geometric.datasets import Planetoid, WebKB, WikipediaNetwork, Amazon, Coauthor, WikiCS
from torch_geometric.utils import add_self_loops, remove_self_loops, to_undirected


PREPROCESS = {
    "cora"      : {"selfloop": 'keep', 'undirected': True , 'norm': True },
    "citeseer"  : {"selfloop": 'keep', 'undirected': True , 'norm': True },
    "pubmed"    : {"selfloop": 'keep', 'undirected': True , 'norm': True },
    "cornell"   : {"selfloop": 'keep', 'undirected': True , 'norm': False},
    "texas"     : {"selfloop": 'keep', 'undirected': True , 'norm': False},
    "wisconsin" : {"selfloop": 'keep', 'undirected': True , 'norm': False},
    "chameleon" : {"selfloop": 'keep', 'undirected': True , 'norm': True },
    "squirrel"  : {"selfloop": 'keep', 'undirected': True , 'norm': True },
    "computers" : {"selfloop": 'keep', 'undirected': False, 'norm': False},
    "photo"     : {"selfloop": 'keep', 'undirected': False, 'norm': False},
    "cs"        : {"selfloop": 'keep', 'undirected': False, 'norm': True },
    "physics"   : {"selfloop": 'keep', 'undirected': False, 'norm': True },
    "wikics"    : {"selfloop": 'keep', 'undirected': False, 'norm': False},
    }


SPLIT = {
    "cora"      : {"type": 'fix', "max_split": 1, "start_select": 0},
    "citeseer"  : {"type": 'fix', "max_split": 1, "start_select": 0},
    "pubmed"    : {"type": 'fix', "max_split": 1, "start_select": 0},
    "cornell"   : {"type": 'fix', "max_split": 10, "start_select": 0},
    "texas"     : {"type": 'fix', "max_split": 10, "start_select": 0},
    "wisconsin" : {"type": 'fix', "max_split": 10, "start_select": 0},
    "chameleon" : {"type": 'fix', "max_split": 10, "start_select": 0},
    "squirrel"  : {"type": 'fix', "max_split": 10, "start_select": 0},
    "computers" : {"type": 'random', 'n_train': 20, 'n_val': 30, 'start_seed': 42},
    "photo"     : {"type": 'random', 'n_train': 20, 'n_val': 30, 'start_seed': 42},
    "cs"        : {"type": 'random', 'n_train': 20, 'n_val': 30, 'start_seed': 42},
    "physics"   : {"type": 'random', 'n_train': 20, 'n_val': 30, 'start_seed': 42},
    "wikics"    : {"type": 'random', 'n_train': 20, 'n_val': 30, 'start_seed': 42},
    }


class _NodeDataset:
    def __init__(self, dataset, preprocess_dict):
        self.dataset_name = dataset
        self.dataset_root = 'MyDataset/tmp'
        self.load()
        self.check()
        self.preprocess(preprocess_dict)


    def load(self):
        name = self.dataset_name
        if name in ['cora', 'citeseer', 'pubmed']:
            self.load_planetoid(name)
        elif name in ['cornell', 'texas', 'wisconsin']:
            self.load_WebKB(name)
        elif name in ['chameleon', 'squirrel']:
            self.load_WikipediaNetwork(name)
        elif name in ['computers', 'photo']:
            self.load_AmazonCoBuy(name)
        elif name in ['cs', 'physics']:
            self.load_CoAuthor(name)
        elif name in ['wikics']:
            self.load_Wiki_CS()
        else:
            raise NotImplementedError(f"{name}")
    

    def preprocess(self, preprocess_dict):
        if preprocess_dict['norm']:
            norm_x = self.x / self.x.sum(dim=1)[:, None]
            if norm_x.isnan().any():
                print("Warning: nan value exist in normalize x, replace with 0")
                norm_x = torch.nan_to_num(norm_x)
            self.x = norm_x
        if preprocess_dict['selfloop'] == 'add':
            self.edge_index = add_self_loops(self.edge_index)[0]
        elif preprocess_dict['selfloop'] == 'remove':
            self.edge_index = remove_self_loops(self.edge_index)[0]
        elif preprocess_dict['selfloop'] != 'keep':
            raise ValueError("selfloop paramater must in [add, remove, keep]")
        if preprocess_dict['undirected']:
            self.edge_index = to_undirected(self.edge_index)


    def load_planetoid(self, name):
        data = Planetoid(root=self.dataset_root, name=name)
        self.unpack(data)
        self.fix_split = {"train_mask": data.train_mask,
                          "val_mask": data.val_mask,
                          "test_mask": data.test_mask}


    def load_WebKB(self, name):
        data = WebKB(root=self.dataset_root, name=name)
        self.unpack(data)
        self.fix_split = {"train_mask": data.train_mask,
                          "val_mask": data.val_mask,
                          "test_mask": data.test_mask}
    

    def load_WikipediaNetwork(self, name):
        data = WikipediaNetwork(root=self.dataset_root, name=name)
        self.unpack(data)
        self.fix_split = {"train_mask": data.train_mask,
                          "val_mask": data.val_mask,
                          "test_mask": data.test_mask}
    

    def load_AmazonCoBuy(self, name):
        data = Amazon(root=self.dataset_root, name=name)
        self.unpack(data)
        self.fix_split = {}


    def load_CoAuthor(self, name):
        data = Coauthor(root=self.dataset_root, name=name)
        self.unpack(data)
        self.fix_split = {}


    def load_Wiki_CS(self):
        data = WikiCS(root=self.dataset_root+'/WikiCS', is_undirected=False)
        self.unpack(data)
        self.fix_split = {}


    def unpack(self, data):
        self.x = data.x
        self.edge_index = data.edge_index
        self.y = data.y
        self.nfeat = self.x.shape[1]
        self.nclass = self.y.max().item() + 1
        self.nnode = self.x.shape[0]


    def check(self):
        if self.x.isnan().sum() > 0:
            raise RuntimeError("Warning: Nan value in x")


class NodeDataset(_NodeDataset):
    def __init__(self, name, device=None, verbose=True, **kwargs):
        self.device = device
        prep_dict, split_dict = self.make_dict(name, kwargs)
        super().__init__(name, prep_dict)
        self.prep_dict = prep_dict
        self.split_dict = split_dict
        self.generate_split()
        if device is not None:
            self.to(device)
        if verbose:
            self.print_state()


    def make_dict(self, name, args_dict):
        prep_dict = {}
        split_dict = {}
        for key, value in args_dict.items():
            if key in ['selfloop', 'undirected', 'norm']:
                prep_dict[key] = value
            if key in ["type", "n_train", "n_val", "start_seed", "start_select"]:
                split_dict[key] = value

        self.n_trial = args_dict["n_trial"] if "n_trial" in args_dict else 1

        default_prep = PREPROCESS[name]
        default_split = SPLIT[name]

        for key in ['selfloop', 'undirected', 'norm']:
            if key not in prep_dict:
                prep_dict[key] = default_prep[key]

        if not bool(split_dict):
            split_dict = default_split

        assert "type" in split_dict, \
            "Please specify a split method."
        if split_dict["type"] == "fix":
            assert "start_select" in split_dict, \
                "split config error."
        if split_dict["type"] == "random":
            assert all(key in split_dict for key in ['start_seed', 'n_train', 'n_val']),\
                "split config error."
        return prep_dict, split_dict


    def generate_split(self):
        split_list = []
        s_type = self.split_dict["type"]
        if s_type == "fix":
            start = self.split_dict["start_select"]
            maximum = self.split_dict["max_split"]
            idx_list = list(range(start, start + self.n_trial))
            assert idx_list[-1] < maximum, "idx error."
            for idx in idx_list:
                split_list.append(self.__fix_split(idx))
        elif s_type == "random":
            n_train = self.split_dict["n_train"]
            n_val = self.split_dict["n_val"]
            seed = self.split_dict["start_seed"]
            seed_list = list(range(seed, seed + self.n_trial))
            for s in seed_list:
                split_list.append(self.__random_split(n_train, n_val, s))
        else:
            raise NotImplementedError(f"{s_type}")
        if self.n_trial == 1:
            self.train_idx, self.val_idx, self.test_idx = split_list[0]
        else:
            self.train_idx, self.val_idx, self.test_idx = [], [], []
            for (train, val, test) in split_list:
                self.train_idx.append(train)
                self.val_idx.append(val)
                self.test_idx.append(test)


    def __fix_split(self, idx):
        name = self.dataset_name
        if name in ["cora", "citeseer", "pubmed"]:
            train = self.fix_split["train_mask"].nonzero().reshape(-1)
            val = self.fix_split["val_mask"].nonzero().reshape(-1)
            test = self.fix_split["test_mask"].nonzero().reshape(-1)
        elif name in ['cornell', 'texas', 'wisconsin', 'chameleon', 'squirrel']:
            train = self.fix_split["train_mask"].T[idx].nonzero().reshape(-1)
            val = self.fix_split["val_mask"].T[idx].nonzero().reshape(-1)
            test = self.fix_split["test_mask"].T[idx].nonzero().reshape(-1)
        else:
            raise NotImplementedError(f"{name}")
        return train, val, test


    def __random_split(self, n_train, n_val, seed):
        torch.manual_seed(seed)
        train_list, val_list, test_list = [], [], []
        for i in range(self.nclass):
            inclass_idx = (self.y == i).nonzero().reshape(-1)
            shuffle = torch.randperm(inclass_idx.shape[0])
            inclass_idx = inclass_idx[shuffle]
            train_list.append(inclass_idx[:n_train])
            val_list.append(inclass_idx[n_train:n_train+n_val])
            test_list.append(inclass_idx[n_train+n_val:])
        train = torch.cat(train_list)
        val = torch.cat(val_list)
        test = torch.cat(test_list)
        return train, val, test


    def to(self, device):
        if device is None: return None
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        self.y = self.y.to(device)
        if self.n_trial == 1:
            self.train_idx = self.train_idx.to(device) 
            self.val_idx = self.val_idx.to(device)
            self.test_idx = self.test_idx.to(device)
        else:
            new_train, new_val, new_test = [], [], []
            for train, val, test in zip(self.train_idx, self.val_idx, self.test_idx):
                new_train.append(train.to(device))
                new_val.append(val.to(device))
                new_test.append(test.to(device))
            self.train_idx = new_train
            self.val_idx = new_val
            self.test_idx = new_test


    def print_state(self):
        norm = self.prep_dict["norm"]
        selfloop = self.prep_dict["selfloop"]
        undirected = self.prep_dict["undirected"]
        s_type = self.split_dict["type"]
        print(f"#####  {self.dataset_name}  #####")
        print(f"[Node] num: {self.nnode}")
        print(f"[Features] dim: {self.nfeat}, norm: {norm}")
        print(f"[Edges] num: {self.edge_index.shape[1]}, selfloop: {selfloop}, undirected: {undirected}")
        print(f"[Class] num: {self.nclass}")
        print(f"[Split] type: {s_type}, num trial: {self.n_trial}")
        if s_type == "random":
            seed = self.split_dict["start_seed"]
            n_train = self.split_dict["n_train"]
            n_val = self.split_dict["n_val"]
            print(f"[Random Split] begin seed: {seed}, num train: {n_train} per class, nval: {n_val} per class")
        if s_type =="fix":
            split_idx = self.split_dict["start_select"]
            print(f"[Fix Split] begin index: {split_idx}")


if __name__ == "__main__":
    data = NodeDataset('cora', device="cuda", type="random", start_seed=41, n_train=50, n_val=50, n_trial=10)
    print("hello")
