import dgl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv


def initialize_dgl(train):
    edges = []
    for src, dst in zip(train["src_id"], train["dst_id"]):
        edges.append((src, dst))
    G = dgl.graph(edges)
    dgl.add_self_loop(G)
    return G

#TODO: try a GCN embedding over random walks.
#class GCN(nn.Module):
#    def __init__(self, in_feats, h_feats, num_classes):
#        super(GCN, self).__init__()
#        self.conv1 = GraphConv(in_feats, h_feats)
#        self.conv2 = GraphConv(h_feats, num_classes)

#    def forward(self, g, in_feat):
#        h = self.conv1(g, in_feat)
#        h = F.relu(h)
#        h = self.conv2(g, h)
#        return h

# Create the model with given dimensions
#model = GCN(g.ndata['feat'].shape[1], 16, dataset.num_classes)
