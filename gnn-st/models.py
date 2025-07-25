import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LSTM
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv, APPNP, GatedGraphConv, TransformerConv, TAGConv, DNAConv, ChebConv

#from torch_geometric.nn import global_mean_pool, DenseDiffPool



class DAGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, K=10, dropout=0.3):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.dagnn = DAGNNConv(K)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.dagnn(x, edge_index)
        return x


class DiffPoolGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.3):
        super().__init__()
        self.gnn1_embed = GCNConv(in_channels, hidden_channels)
        self.gnn1_pool = GCNConv(in_channels, hidden_channels)

        self.gnn2_embed = GCNConv(hidden_channels, hidden_channels)
        self.gnn2_pool = GCNConv(hidden_channels, hidden_channels)

        self.lin = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        s = F.softmax(self.gnn1_pool(x, edge_index), dim=-1)
        x = F.relu(self.gnn1_embed(x, edge_index))

        s = F.softmax(self.gnn2_pool(x, edge_index), dim=-1)
        x = F.relu(self.gnn2_embed(x, edge_index))

        x = global_mean_pool(x, batch)
        return self.lin(x)


class RecurrentGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.3):
        super().__init__()
        self.gcn = GCNConv(in_channels, hidden_channels)
        self.lstm = LSTM(hidden_channels, hidden_channels, num_layers, batch_first=True)
        self.lin = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.gcn(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x_seq = x.unsqueeze(1)  # shape: [N, 1, F]
        x_lstm, _ = self.lstm(x_seq)
        x = self.lin(x_lstm[:, -1, :])
        return x


class ChebNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, K=3, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(ChebConv(in_channels, hidden_channels, K=K))
        for _ in range(num_layers - 2):
            self.convs.append(ChebConv(hidden_channels, hidden_channels, K=K))
        self.convs.append(ChebConv(hidden_channels, out_channels, K=K))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)

class DNANet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.3):
        super().__init__()
        self.lin = nn.Linear(in_channels, hidden_channels)
        self.dna = DNAConv(hidden_channels, heads=heads)
        self.lin_out = nn.Linear(hidden_channels * heads, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.lin(x).unsqueeze(1)  # shape: [N, 1, F]
        xs = [x.squeeze(1)]
        for _ in range(3):  # number of propagation steps
            x = self.dna(xs, edge_index)
            xs.append(x)
        x = self.lin_out(x)
        return x

class TAGNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, K=3, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(TAGConv(in_channels, hidden_channels, K=K))
        for _ in range(num_layers - 2):
            self.convs.append(TAGConv(hidden_channels, hidden_channels, K=K))
        self.convs.append(TAGConv(hidden_channels, out_channels, K=K))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)

class TransformerGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, heads=4, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(TransformerConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        for _ in range(num_layers - 2):
            self.convs.append(TransformerConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout))
        self.convs.append(TransformerConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)

class ResGatedGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.3):
        super().__init__()
        self.lin_in = nn.Linear(in_channels, hidden_channels)
        self.conv = GatedGraphConv(out_channels=hidden_channels, num_layers=num_layers)
        self.lin_out = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.lin_in(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv(x, edge_index)
        x = self.lin_out(x)
        return x


# ========= GNN Model Classes =========
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList([GCNConv(in_channels, hidden_channels)])
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)

class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList([SAGEConv(in_channels, hidden_channels)])
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)

class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.3, heads=1):
        super().__init__()
        self.convs = nn.ModuleList([GATConv(in_channels, hidden_channels, heads=heads)])
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads))
        self.convs.append(GATConv(hidden_channels * heads, out_channels, heads=1))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.elu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)

class GIN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        nn1 = nn.Sequential(nn.Linear(in_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))
        self.convs.append(GINConv(nn1))
        for _ in range(num_layers - 2):
            nnx = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))
            self.convs.append(GINConv(nnx))
        nnf = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, out_channels))
        self.convs.append(GINConv(nnf))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)

class APPNPNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.3, K=10, alpha=0.1):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.prop = APPNP(K, alpha)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return self.prop(x, edge_index)

class SGCNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.3, K=2):
        super().__init__()
        self.conv = SGC(in_channels, out_channels, K=K)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv(x, edge_index)

# ========= Model Selection Dict =========
MODEL_CLASSES = {
    'gcn': GCN,
    'sage': GraphSAGE,
    'gat': GAT,
    'gin': GIN,
    'appnp': APPNPNet,
    'recurrent': RecurrentGNN,
    'cheb': ChebNet,
    'dna': DNANet,
    'tag': TAGNet,
    'transformer': TransformerGNN,
    'gated': ResGatedGNN,
}
