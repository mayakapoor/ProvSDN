import dgl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, out_feats, n_layers):
        super(GCN, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.h_feats = h_feats
        self.n_layers = n_layers
        self.layer = GraphConv(h_feats, h_feats, weight=True, bias=True)
        self.final_layer = GraphConv(h_feats, out_feats, weight=True, bias=True)

    def node_feat(self):
        node_onehot = np.eye(self.in_feats, self.h_feats)
        return torch.from_numpy(node_onehot).float()

    def forward(self, g):
        l = self.n_layers
        h = F.relu(self.layer(g, self.node_feat()))
        while (l > 1):
            h = F.relu(self.layer(g, h))
            l = l - 1
        return self.final_layer(g, h)
