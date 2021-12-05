import dgl
import dgl.function as fn
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeLayer(nn.Module):
    def __init__(self, n_edges, h_feats, encoding_method):
        super(EdgeLayer, self).__init__()
        self.encoding = encoding_method
        self.W1 = nn.Linear(n_edges, h_feats)
        self.W2 = nn.Linear(h_feats, 1)

    def apply_edges(self, edges):
        """
        Computes a scalar score for each edge of the given graph.

        Parameters
        ----------
        edges :
            Has three members ``src``, ``dst`` and ``data``, each of
            which is a dictionary representing the features of the
            source nodes, the destination nodes, and the edges
            themselves.

        Returns
        -------
        dict
            A dictionary of new edge features.
        """
        if self.encoding == 'Average':
            h = (edges.src['h'] + edges.dst['h']) / 2
        elif self.encoding == 'Hadamard':
            h = edges.src['h'] * edges.dst['h']
        elif self.encoding == 'WeightedL1':
            h = abs(edges.src['h'] - edges.dst['h'])
        elif self.encoding == 'WeightedL2':
            h = (edges.src['h'] - edges.dst['h']) ** 2
        return {'score': h}

    def forward(self, g, node_features, edge_features):
        with g.local_scope():
            edge_features = torch.cat([edge_features, torch.zeros([g.num_nodes(), 13])], 0)
            g.add_nodes(len(node_features) - g.num_nodes())
            g.ndata['h'] = node_features
            g.edata['h'] = edge_features
            g.apply_edges(self.apply_edges)
            return g.edata['score']
