import dgl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn.linear_model as lm
import sklearn.metrics as skm
from dgl.nn import SAGEConv
import dgl.function as fn

class SAGE(nn.Module):
    def __init__(self, in_feats, h_feats, n_classes, n_layers):
        super(SAGE, self).__init__()
        self.n_nodes = in_feats
        self.n_hidden = h_feats
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.layer = SAGEConv(in_feats=h_feats, out_feats=h_feats, aggregator_type='mean')

    def node_feat(self):
        node_feat = torch.ones(self.n_nodes, self.n_hidden)
        return node_feat

    def forward(self, g, in_feat):
        l = self.n_layers
        h = F.relu(self.layer(g, in_feat))
        l = l - 1
        while (l > 1):
            h = F.relu(self.layer(g, h))
            l = l - 1
        return self.layer(g, h)
