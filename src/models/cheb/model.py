import dgl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import ChebConv

class Cheb(nn.Module):
    def __init__(self, in_feats, h_feats, out_feats, n_layers):
        super(Cheb, self).__init__()
        self.in_feats = in_feats
        self.out_feats = n_classes
        self.h_feats = h_feats
        self.n_layers = n_layers
        self.layer = ChebConv(h_feats, h_feats, 2)
        self.final_layer = ChebConv(h_feats, out_feats, 2)

    def node_feat(self):
        node_onehot = np.eye(self.in_feats, self.h_feats)
        return torch.from_numpy(node_onehot).float()

    def forward(self, g):
        l = self.n_layers
        h = F.relu(self.layer(g, features))
        while (l > 1):
            h = F.relu(self.layer(g, h))
            l = l - 1
        return self.final_layer(g, h)
