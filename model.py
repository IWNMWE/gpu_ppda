import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import GCNConv

class GCNet(nn.Module):
    def __init__(self, n_classes, input_dim=1433, embd_dim=16):
        super(GCNet, self).__init__()
        self.conv1 = GCNConv(input_dim, embd_dim)
        self.conv2 = GCNConv(embd_dim, embd_dim)
        self.conv2 = GCNConv(embd_dim, n_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
    
def coragcn(n_classes, input_dim=1433):
    return GCNet(n_classes=n_classes, input_dim=input_dim)

def niidgcn(n_classes):
    return GCNet(n_classes, input_dim=40)