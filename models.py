import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, ChebConv

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index, batch, use_pooling=True):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        
        if use_pooling:
            x = global_mean_pool(x, batch)
        
        x = F.dropout(x, p=0.5, training=self.training)
        return F.log_softmax(x, dim=1)
    
    
class ChebNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, K=5):
        super(ChebNet, self).__init__()
        self.conv1 = ChebConv(in_channels, hidden_channels,  K)
        self.conv2 = ChebConv(hidden_channels, hidden_channels, K)
        self.conv3 = ChebConv(hidden_channels, out_channels, K)
    
    def forward(self, x, edge_index, batch, use_pooling=True):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        
        if use_pooling:
            x = global_mean_pool(x, batch)
        
        x = F.dropout(x, p=0.5, training=self.training)
        return F.log_softmax(x, dim=1)