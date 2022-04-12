import torch
import torch.nn as nn

import torch
import torch.nn as nn
from random import random
from torch_geometric.nn import GCNConv, GraphConv
from torch_geometric.nn import global_max_pool, global_add_pool, global_mean_pool
    
    
class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hid_dim, improved=True)
        self.conv2 = GCNConv(hid_dim, hid_dim * 2, improved=True)
        self.conv3 = GCNConv(hid_dim * 2, hid_dim, improved=True)

    def forward(self, data, edge_index, batch):
        # batch = [num node * num graphs in batch]
        # data = [num node * num graphs in batch, num features]
        # edge_index = [2, num edges * num graph in batch]
        # Obtain node embeddings 
        # data = [num node * num graphs in batch, hid dim]
        output = self.conv1(data, edge_index)
        output = output.relu()
        output = self.conv2(output, edge_index)
        output = output.relu()
        output = self.conv3(output, edge_index)

        # Readout layer
        # output = [batch size, hid dim]
        output = global_max_pool(output, batch)
        
        return output

class GraphClassifier(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim, device):
        super().__init__()
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.device = device

        self.encoder = Encoder(self.input_dim, self.hid_dim)
        self.fc1 = nn.Linear(self.hid_dim, 2 * self.hid_dim)
        self.fc2 = nn.Linear(2 * self.hid_dim, self.output_dim)
        self.device = device

    def forward(self, data):
        features = self.encoder(data.x, data.edge_index, data.batch)
        output = self.fc1(features)
        output = output.relu()
        output = self.fc2(output)

        return output
