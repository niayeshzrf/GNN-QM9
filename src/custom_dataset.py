import os
import torch
from torch_geometric.data import InMemoryDataset, Data

class QM9ProcessedDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        self.root = root
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        # Return a dummy file to satisfy PyG
        return ['qm9_dummy.txt']
    
    @property
    def processed_file_names(self):
        return ['qm9_graphs.pt']
    
    def download(self):
        # No need to download - we already did that manually
        pass

    def process(self):
        # Load your list of Data objects (from qm9_loader.py)

        from load_qm9 import get_graph_data_list
        data_list = get_graph_data_list()

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
                                                
