import dgl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, n_classes, n_layers):
        super(GCN, self).__init__()
        self.layer = GraphConv(h_feats, h_feats, weight=True, bias=True)
        self.n_nodes = in_feats
        self.n_classes = n_classes
        self.n_hidden = h_feats
        self.n_layers = n_layers

    def node_feat(self):
        node_feat = torch.ones(self.n_nodes, self.n_hidden)
        return node_feat

    def forward(self, g, features):
        l = self.n_layers
        h = F.relu(self.layer(g, features))
        l = l - 1
        while (l > 1):
            h = F.relu(self.layer(g, h))
            l = l - 1
        return h
