import dgl
import numpy as np
import pandas as pd
import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dgl.nn.pytorch import SumPooling, AvgPooling, MaxPooling
from dgl.utils import expand_as_pair
import dgl.function as fn

class Rahmen(nn.Module):
    def __init__(self, relations, in_feats, h_feats, n_classes, n_layers):
        super(Rahmen, self).__init__()
        self.n_nodes = in_feats
        self.n_classes = n_classes
        self.n_hidden = h_feats
        self.n_layers = n_layers
        self.relations = relations
        agg_type='mean'

        self.transform = nn.ModuleDict({
            (str(src) + "  " + str(dst)): MessageTransform(
                in_dim=self.n_nodes,
                out_dim=self.n_hidden,
                dropout=0,
                activation='relu',
                norm=True
            )
            for src, dst in zip(relations[0].numpy(), relations[1].numpy())
        })



        self.attention = SemanticAttention(len(relations), self.n_hidden, 16)
        # TODO: Separate node reduce and global readout functions?
        self.reduce_fn, self.readout_fn = self._get_reduce_fn(agg_type)

    @staticmethod
    def _get_reduce_fn(agg_type):
        if agg_type == 'mean':
            reduce_fn = fn.mean
            readout_fn = AvgPooling()
        elif agg_type == 'max':
            reduce_fn = fn.max
            readout_fn = MaxPooling()
        elif agg_type == 'sum':
            reduce_fn = fn.sum
            readout_fn = SumPooling()
        else:
            raise ValueError('Invalid aggregation function')

        return reduce_fn, readout_fn

    def forward(self, graph, feat):
        h = torch.zeros(len(self.relations), graph.num_nodes(), self.n_hidden)
        with graph.local_scope():
            feat_src, feat_dst = expand_as_pair(feat, graph)
            for i, rel in enumerate(self.relations):
                if rel in graph.etypes:
                    graph.srcdata['h'] = feat_src
                    # TODO: Add node-level attention from GAT
                    graph.update_all(
                        fn.copy_u('h', 'm'),
                        self.reduce_fn('m', 'neigh'),
                        etype=rel
                    )

                    h_rel = feat_dst + graph.dstdata['neigh']

                    h[i] = self.transform[rel](h_rel)

            h = self.attention(graph, h)

        return self.readout_fn(graph, h)

    def node_feat(self):
        node_feat = torch.ones(self.n_nodes, self.n_hidden)
        return node_feat

class SemanticAttention(nn.Module):
    def __init__(self, num_relations, in_dim, dim_a, out_dim=1, dropout=0.):
        super(SemanticAttention, self).__init__()
        self.num_relations = num_relations
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dim_a = dim_a
        self.dropout = nn.Dropout(dropout)

        self.weights_s1 = nn.Parameter(
            torch.FloatTensor(self.num_relations, self.in_dim, self.dim_a)
        )
        self.weights_s2 = nn.Parameter(
            torch.FloatTensor(self.num_relations, self.dim_a, self.out_dim)
        )

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('tanh')
        nn.init.xavier_uniform_(self.weights_s1.data, gain=gain)
        nn.init.xavier_uniform_(self.weights_s2.data)

    def forward(self, graph, h):
        # Shape of input h: (num_relations, num_nodes, dim)
        # Output shape: (num_nodes, dim)
        graph.ndata['h'] = torch.zeros(graph.num_nodes(), h.size(-1), device=graph.device)
        attention = F.softmax(
            torch.matmul(
                torch.tanh(
                    torch.matmul(h, self.weights_s1)
                ),
                self.weights_s2
            ),
            dim=0
        ).squeeze()

        attention = self.dropout(attention)

        # TODO: FFT option: https://pytorch.org/docs/stable/generated/torch.fft.rfft2.html
        graph.ndata['h'] = torch.einsum('rb,rbd->bd', attention, h)

        return graph.ndata.pop('h')

class MessageTransform(nn.Module):
    def __init__(
            self,
            in_dim,
            out_dim,
            num_layers=2,
            dropout=0.,
            activation='relu',
            norm=True,
    ):
        super(MessageTransform, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.dropout = nn.Dropout(dropout)
        self.activation = self._get_activation_fn(activation)
        self.norm = nn.LayerNorm(self.out_dim, elementwise_affine=True) if norm else None

        self.layers = nn.ModuleList([
            nn.Linear(self.in_dim, self.out_dim) if i < num_layers-1
            else nn.Linear(self.out_dim, self.out_dim)
            for i in range(num_layers)
        ])

    @staticmethod
    def _get_activation_fn(activation):
        if activation is None:
            act_fn = None
        elif activation == 'relu':
            act_fn = F.relu
        elif activation == 'elu':
            act_fn = F.elu
        elif activation == 'gelu':
            act_fn = F.gelu
        else:
            raise ValueError('Invalid activation function.')

        return act_fn

    def forward(self, x):
        for layer in self.layers:
            x = self.dropout(x)
            x = layer(x)

            if self.norm:
                # TODO: LayerNorm broken. Fix dimensions
                x = self.norm(x)
            if self.activation:
                x = self.activation(x)

        return x
