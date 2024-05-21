import torch
import torch_geometric.transforms as T
from torch_geometric.utils.convert import to_scipy_sparse_matrix
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from pytorch_lightning import seed_everything


class RemoveEdgeAttr(object):
    def __call__(self, data):
        if data.edge_attr is not None:
            data.edge_attr = None
        if data.x is None:
            if hasattr(data, 'adj'):
                data.x = data.adj.sum(1).view(-1, 1)
            else:
                adj = to_scipy_sparse_matrix(data.edge_index).sum(1)
                data.x = torch.FloatTensor(adj.sum(1)).view(-1, 1)

        data.y = data.y.squeeze(0)
        data.x = data.x.float()
        return data

class ConcatPos(object):
    def __call__(self, data):
        if data.edge_attr is not None:
            data.edge_attr = None
        data.x = torch.cat([data.x, data.pos], dim=1)
        data.pos = None
        return data

class Complete(object):
    def __call__(self, data):
        if data.edge_attr is not None:
            data.edge_attr = None
        if data.x is None:
            if hasattr(data, 'adj'):
                data.x = data.adj.sum(1).view(-1, 1)
            else:
                adj = to_scipy_sparse_matrix(data.edge_index).sum(1)
                data.x = torch.FloatTensor(adj.sum(1)).view(-1, 1)
        return data
    

class _GraphDataset:
    def __init__(self, dataset):
        self.dataset_name = dataset
        self.dataset_root = 'MyDataset/tmp'
        self.load()
    
    def load(self):
        name = self.dataset_name
        if name in ['DD', 'MUTAG', 'NCI1', 'PROTEINS', 'ENZYMES', 'NCI109']:
            data = TUDataset(self.dataset_root, name=name, 
                             transform=T.Compose([Complete()]), use_node_attr=True)
            self.unpack(data)
        else:
            raise NotImplementedError(f"{name}")

    def unpack(self, data):
        self.data = data
        self.ngraph = len(data)
        self.nnode = data.data.x.shape[0]
        self.nfeat = data.data.x.shape[1]
        self.nclass = data.data.y.max().item() + 1


class GraphDataset(_GraphDataset):
    def __init__(self, dataset, verbose=True):
        super().__init__(dataset)
        if verbose:
            self.print_state()
    
    def random_split_loader(self, batch_size, proportion, seed=0):
        deno = sum(proportion)
        n_train = int(self.ngraph * proportion[0] / deno)
        n_val = int(self.ngraph * proportion[1] / deno)
        seed_everything(seed)
        data = self.data.shuffle()
        train_loader = DataLoader(data[:n_train], batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(data[n_train:n_train+n_val], batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(data[n_train+n_val:], batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, test_loader

    def print_state(self):
        pass

if __name__ == "__main__":
    d = GraphDataset('NCI109')
    print("hello")