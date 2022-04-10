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
        self.conv3 = GCNConv(hid_dim * 2, hid_dim * 4, improved=True)
        self.conv4 = GCNConv(hid_dim * 4, hid_dim * 2, improved=True)
        self.conv5 = GCNConv(hid_dim * 2, hid_dim, improved=True)
        self.drop1 = nn.Dropout(0.9)

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
        output = output.relu()
        output = self.conv4(output, edge_index)
        output = output.relu()
        output = self.conv5(output, edge_index)
        output = self.drop1(output)

        # Readout layer
        # output = [batch size, hid dim]
        output = global_add_pool(output, batch)
        
        return output

class GraphClassifier(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim, device):
        super().__init__()
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.device = device

        self.encoder = Encoder(self.input_dim, self.hid_dim)
        self.fc_out = nn.Linear(self.hid_dim, self.output_dim)
        self.device = device

    def forward(self, data):
        features = self.encoder(data.x, data.edge_index, data.batch).unsqueeze(0)
        output = self.fc_out(features)

        return output
