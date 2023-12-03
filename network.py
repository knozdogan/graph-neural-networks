import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GraphNetwork(torch.nn.Module):
    def __init__(self, embedding=64, num_features=3):
        super().__init__()
        self.initial_conv = GCNConv(num_features, embedding)
        self.conv1 = GCNConv(embedding, embedding)
        self.conv2 = GCNConv(embedding, embedding//2)
        self.conv3 = GCNConv(embedding//2, embedding//2)
        self.conv4 = GCNConv(embedding//2, embedding//2)
        self.conv5 = GCNConv(embedding//2, 1)

    def forward(self, x, edge_index):

        hidden = self.initial_conv(x, edge_index)
        hidden = F.elu(hidden)

        hidden = self.conv1(hidden, edge_index)
        hidden = F.elu(hidden)

        hidden = self.conv2(hidden, edge_index)
        hidden = F.elu(hidden)

        hidden = self.conv3(hidden, edge_index)
        hidden = F.elu(hidden)

        hidden = self.conv4(hidden, edge_index)
        hidden = F.elu(hidden)

        out = self.conv5(hidden, edge_index)

        return out