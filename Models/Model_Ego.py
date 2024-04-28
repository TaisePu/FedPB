import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.nn.conv.sage_conv import SAGEConv
import torch.nn.functional as F


class LowLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(LowLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.init_params()

    def init_params(self):
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.kaiming_normal_(param)

    def forward(self, x):
        x = self.linear(x)
        x = F.relu(x)
        return x


class GNNLayer(nn.Module):
    def __init__(self, sageMode, in_feats, out_feats, h_feats, linear, layer_size=2):
        super(GNNLayer, self).__init__()
        self.layer_size = layer_size
        self.sageMode = sageMode
        self.linear = linear

        if self.sageMode == "GraphSAGE":
            self.sage1 = SAGEConv(in_feats, h_feats)
            self.sage2 = SAGEConv(h_feats, out_feats)
            self.sagex = [SAGEConv(h_feats, h_feats)
                          for layer in range(layer_size - 2)]
        elif self.sageMode == "GAT":
            self.sage1 = GATConv(in_feats, h_feats, dropout=0.1)
            self.sage2 = GATConv(h_feats, out_feats, dropout=0.1)
            self.sagex = [GATConv(h_feats, h_feats, dropout=0.1)
                          for layer in range(layer_size - 2)]
        self.init_params()

    def init_params(self):
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.kaiming_normal_(param)

    def forward(self, x, edge_index):
        x = self.sage1(x, edge_index)
        if not self.linear:
            x = F.relu(x)
        for layer in range(self.layer_size - 2):
            x = self.sagex[layer](x, edge_index)
            if not self.linear:
                x = F.relu(x)
        x = self.sage2(x, edge_index)
        x = F.relu(x)
        return x


class Classification(nn.Module):
    def __init__(self, in_feats, num_classes):
        super(Classification, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_feats))
        self.init_params()

    def init_params(self):
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.kaiming_normal_(param)

    def forward(self, embeds):
        logists = torch.log_softmax(self.weight.mm(embeds.t()).t(), 1)
        return logists

class MyModel(nn.Module):
    def __init__(self, sageMode, in_feats, h_feats, out_feats,
                 linear, num_classes, layer_size=2):
        super(MyModel, self).__init__()
        self.layer_size = layer_size
        self.sageMode = sageMode
        self.linear = linear
        self.linear = nn.Linear(in_feats, h_feats)

        if self.sageMode == "GraphSAGE":
            self.sage1 = SAGEConv(h_feats, h_feats)
            self.sage2 = SAGEConv(h_feats, h_feats)
            self.sagex = [SAGEConv(h_feats, h_feats) for layer in range(layer_size - 2)]
        elif self.sageMode == "GAT":
            self.sage1 = GATConv(h_feats, h_feats, dropout=0.1)
            self.sage2 = GATConv(h_feats, h_feats, dropout=0.1)
            self.sagex = [GATConv(h_feats, h_feats, dropout=0.1) for layer in range(10)]

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, h_feats))
        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.kaiming_normal_(param)

    def forward(self, x, edge_index):
        x = self.linear(x)
        x = F.relu(x)
        '''-------'''
        x = self.sage1(x, edge_index)
        if not self.linear:
            x = F.relu(x)
        for layer in range(self.layer_size - 2):
            x = self.sagex[layer](x, edge_index)
            if not self.linear:
                x = F.relu(x)
        x = self.sage2(x, edge_index)
        x = F.relu(x)
        '''-------'''
        # logists = torch.log_softmax(self.weight.mm(x.t()).t(), 1)
        x = self.weight.mm(x.t()).t()
        return x #logists


class FedGCN(nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super(FedGCN, self).__init__()
        self.linear1 = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, out_feats)
        self.init_params()

    def init_params(self):
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.kaiming_normal_(param)

    def forward(self, x, adj):
        x = self.linear1(x)
        x = adj.mm(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.1)
        x = adj.mm(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = F.softmax(x, dim=1)
        return x


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, num_classes)
        # self.conv3 = GCNConv(32, num_classes)
        # self.norm = torch.nn.BatchNorm1d(64)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        # x = self.norm(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        # x = self.conv3(x, edge_index)
        return x