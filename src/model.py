import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.nn import GINConv, global_mean_pool

class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.2):
        super().__init__()

        self.convs = nn.ModuleList()
        self.convs.append(GINConv(Seq(Lin(in_channels, hidden_channels), ReLU(), Lin(hidden_channels, hidden_channels))))

        for _ in range(num_layers - 2):
            self.convs.append(GINConv(Seq(Lin(hidden_channels, hidden_channels), ReLU(), Lin(hidden_channels, hidden_channels))))

        self.convs.append(GINConv(Seq(Lin(hidden_channels, hidden_channels), ReLU(), Lin(hidden_channels, out_channels))))

        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        x = global_mean_pool(x, batch)
        return x