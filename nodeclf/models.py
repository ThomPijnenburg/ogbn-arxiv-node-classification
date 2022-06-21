import torch.nn as nn
import torch.nn.functional as F


from layers import GCNLayer


class GCN(nn.Module):
    def __init__(self, c_in, c_hidden, c_out, dropout_rate=0.5):
        super().__init__()

        self.gcn1 = GCNLayer(c_in, c_hidden)
        self.gcn2 = GCNLayer(c_hidden, c_hidden)
        self.gcn3 = GCNLayer(c_hidden, c_out)

        self.dropout = dropout_rate

    def forward(self, x, adj):
        x = F.relu(self.gcn1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gcn2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcn3(x, adj)
        return F.log_softmax(x, dim=1)


class MLP(nn.Module):
    def __init__(self, n_layers, c_in, c_hidden, dropout, num_classes):
        super().__init__()
        layers = []

        layers += [nn.Linear(c_in, c_hidden), nn.ReLU(), nn.Dropout(dropout)]

        for i in range(n_layers - 2):
            layers += [
                nn.Linear(c_hidden, c_hidden), nn.ReLU(), nn.Dropout(dropout)
            ]

        layers += [
            nn.Linear(c_hidden, num_classes)
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return F.log_softmax(x, dim=1)
