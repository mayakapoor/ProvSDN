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
    def __init__(self, in_feats, h_feats, out_feats, n_layers):
        super(SAGE, self).__init__()
        self.in_feats = in_feats
        self.h_feats = h_feats
        self.out_feats = out_feats
        self.n_layers = n_layers
        self.layer = SAGEConv(in_feats=h_feats, out_feats=h_feats, aggregator_type='pool')
        self.final_layer = SAGEConv(in_feats=h_feats, out_feats=out_feats, aggregator_type='pool')

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
