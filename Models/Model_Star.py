import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, global_add_pool


class serverGIN(torch.nn.Module):
    def __init__(self, args):
        super(serverGIN, self).__init__()
        self.graph_convs = torch.nn.ModuleList()
        self.nn1 = torch.nn.Sequential(torch.nn.Linear(args.hidden, args.hidden), torch.nn.ReLU(),
                                       torch.nn.Linear(args.hidden, args.hidden))
        self.graph_convs.append(GINConv(self.nn1))

        for l in range(args.nlayer - 1):
            self.nnk = torch.nn.Sequential(torch.nn.Linear(args.hidden, args.hidden), torch.nn.ReLU(),
                                           torch.nn.Linear(args.hidden, args.hidden))
            self.graph_convs.append(GINConv(self.nnk))


class serverGIN_dc(torch.nn.Module):
    def __init__(self, args):
        super(serverGIN_dc, self).__init__()
        # embedding
        self.embedding_s = torch.nn.Linear(args.n_se, args.hidden)
        self.Whp = torch.nn.Linear(args.hidden + args.hidden, args.hidden)

        self.graph_convs = torch.nn.ModuleList()
        self.nn1 = torch.nn.Sequential(torch.nn.Linear(args.hidden + args.hidden, args.hidden), torch.nn.ReLU(),
                                       torch.nn.Linear(args.hidden, args.hidden))
        self.graph_convs.append(GINConv(self.nn1))

        self.graph_convs_s_gcn = torch.nn.ModuleList()
        self.graph_convs_s_gcn.append(GCNConv(args.hidden, args.hidden))

        for l in range(args.nlayer - 1):
            self.nnk = torch.nn.Sequential(torch.nn.Linear(args.hidden + args.hidden, args.hidden), torch.nn.ReLU(),
                                           torch.nn.Linear(args.hidden, args.hidden))
            self.graph_convs.append(GINConv(self.nnk))
            self.graph_convs_s_gcn.append(GCNConv(args.hidden, args.hidden))


class GIN(torch.nn.Module):
    def __init__(self, args, num_feature, nclass):
        super(GIN, self).__init__()
        self.num_layers = args.nlayer
        self.dropout = args.dropout

        self.pre = torch.nn.Sequential(torch.nn.Linear(num_feature, args.hidden))

        self.graph_convs = torch.nn.ModuleList()
        self.nn1 = torch.nn.Sequential(torch.nn.Linear(args.hidden, args.hidden),
                                       torch.nn.ReLU(), torch.nn.Linear(args.hidden, args.hidden))
        self.graph_convs.append(GINConv(self.nn1))

        for l in range(args.nlayer - 1):
            self.nnk = torch.nn.Sequential(torch.nn.Linear(args.hidden, args.hidden),
                                           torch.nn.ReLU(), torch.nn.Linear(args.hidden, args.hidden))
            self.graph_convs.append(GINConv(self.nnk))

        self.post = torch.nn.Sequential(torch.nn.Linear(args.hidden, args.hidden), torch.nn.ReLU())
        self.readout = torch.nn.Sequential(torch.nn.Linear(args.hidden, nclass))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.pre(x)
        for i in range(len(self.graph_convs)):
            x = self.graph_convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)

        x = global_add_pool(x, batch)
        x = self.post(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.readout(x)
        x = F.log_softmax(x, dim=1)

        return x

    def loss(self, pred, label):
        return F.nll_loss(pred, label)


class GIN_dc(torch.nn.Module):
    def __init__(self, args, num_feature, num_class):
        super(GIN_dc, self).__init__()
        self.num_layers = args.nlayer
        self.dropout = args.dropout
        self.pre = torch.nn.Sequential(torch.nn.Linear(num_feature, args.hidden))
        self.embedding_s = torch.nn.Linear(args.n_se, args.hidden)

        self.graph_convs = torch.nn.ModuleList()
        self.nn1 = torch.nn.Sequential(torch.nn.Linear(args.hidden + args.hidden, args.hidden),
                                       torch.nn.ReLU(), torch.nn.Linear(args.hidden, args.hidden))
        self.graph_convs.append(GINConv(self.nn1))

        self.graph_convs_s_gcn = torch.nn.ModuleList()
        self.graph_convs_s_gcn.append(GCNConv(args.hidden, args.hidden))

        for l in range(args.nlayer - 1):
            self.nnk = torch.nn.Sequential(torch.nn.Linear(args.hidden + args.hidden, args.hidden),
                                           torch.nn.ReLU(), torch.nn.Linear(args.hidden, args.hidden))
            self.graph_convs.append(GINConv(self.nnk))
            self.graph_convs_s_gcn.append(GCNConv(args.hidden, args.hidden))

        self.Whp = torch.nn.Linear(args.hidden + args.hidden, args.hidden)
        self.post = torch.nn.Sequential(torch.nn.Linear(args.hidden, args.hidden), torch.nn.ReLU())
        self.readout = torch.nn.Sequential(torch.nn.Linear(args.hidden, num_class))

    def forward(self, data):
        feature, y, edge_index, batch, s = data.x, data.y, data.edge_index, data.batch, data.stc_enc
        x = self.pre(feature)
        s = self.embedding_s(s)

        for i in range(len(self.graph_convs)):
            x = torch.cat((x, s), -1)
            x = self.graph_convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            s = self.graph_convs_s_gcn[i](s, edge_index)
            s = torch.tanh(s)

        x = self.Whp(torch.cat((x, s), -1))
        x = global_add_pool(x, batch)
        x = self.post(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.readout(x)
        x = F.log_softmax(x, dim=1)

        return x

    def loss(self, pred, label):
        return F.nll_loss(pred, label)


class serverGraphSage(torch.nn.Module):
    def __init__(self, nlayer, nhid):
        super(serverGraphSage, self).__init__()
        self.graph_convs = torch.nn.ModuleList()
        self.graph_convs.append(SAGEConv(nhid, nhid))

        for l in range(nlayer - 1):
            self.graph_convs.append(SAGEConv(nhid, nhid))


class GraphSage(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayer, dropout):
        super(GraphSage, self).__init__()
        self.num_layers = nlayer
        self.dropout = dropout
        self.pre = torch.nn.Sequential(torch.nn.Linear(nfeat, nhid))
        self.graph_convs = torch.nn.ModuleList()
        self.graph_convs.append(SAGEConv(nhid, nhid))

        for l in range(nlayer - 1):
            self.graph_convs.append(SAGEConv(nhid, nhid))

        self.post = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU())
        self.readout = torch.nn.Sequential(torch.nn.Linear(nhid, nclass))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.pre(x)

        for i in range(len(self.graph_convs)):
            x = self.graph_convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)

        x = global_add_pool(x, batch)
        x = self.post(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.readout(x)
        x = F.log_softmax(x, dim=1)

        return x

    def loss(self, pred, label):
        return F.nll_loss(pred, label)














