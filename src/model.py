import models.nw.model as NetWalk
import models.gcn.model as GCN
import models.sage.model as SAGE
import models.cheb.model as Cheb

import process

from classifiers.binary import BinaryClassifier
from dgl.nn.pytorch import SumPooling

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def initialize_dgl(data):
    """
    Returns a DGL graph from pandas dataframe with edge features added.
    """
    G = dgl.DGLGraph()

    #index nodes
    uniq = set()
    for src, dst in zip(data['src_id'], data['dst_id']):
        uniq.add(src)
        uniq.add(dst)

    G.add_edges(data["src_id"].to_numpy(), data["dst_id"].to_numpy())
    G.edata['h'] = process.edge_feat(data)
    G = dgl.add_self_loop(G)
    return G, data


class Model(nn.Module):
    """
    The plug-and-play model which supports the different kinds of embeddings
    """
    def __init__(self, in_feat, h_feat, out_feat, embModel, encoding_method,
                 n_layers, optimizer_type, learning_rate, num_epochs, train, test):
        super(Model, self).__init__()
        self.in_feat = in_feat
        self.h_feat = h_feat
        self.out_feat = out_feat
        self.embModel = embModel
        self.n_layers = n_layers
        self.train_data = train
        self.test_data = test
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.encoding_method = encoding_method

        self.G, self.train_data = initialize_dgl(self.train_data)

        """
        NetWalk does not use DGL yet, so the number of hidden features is different!
        """
        if self.embModel != "NetWalk":
            self.pred = BinaryClassifier(out_feat + 6)
        else:
            self.pred = BinaryClassifier(h_feat + 6)

        """
        Optimizer
        """
        if self.optimizer_type == "adagrad":
            self.optimizer = torch.optim.Adagrad(self.parameters(), lr=self.learning_rate,
                                              initial_accumulator_value=1e-8)
        elif self.optimizer_type == "asgd":
            self.optimizer = torch.optim.ASGD(self.parameters(), lr=self.learning_rate)
        elif optimizer_type == "rmsprop":
            self.optimizer = torch.optim.RMSprop(self.parameters(), lr=self.learning_rate, alpha=0.99, eps=1e-08)
        elif self.optimizer_type == "momentum":
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.95)
        else:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-8)

        """
        One of the supported embedding models
        """
        if self.embModel == "SAGE":
            self.embedder = SAGE.SAGE(in_feat, h_feat, out_feat, n_layers)

        elif self.embModel == "NetWalk":
            self.embedder = NetWalk.NetWalk(train, test, in_feat, h_feat, out_feat)

        elif self.embModel == "GCN":
            self.embedder = GCN.GCN(in_feat, h_feat, out_feat, n_layers)

        elif self.embModel == "Spectral":
            self.embedder = Cheb.Cheb(in_feat, h_feat, out_feat, n_layers)

        #elif self.embModel == "EdgeConv":
        #    self.embedder = Edge.Edge(in_feat, h_feat, out_feat, n_layers)
        #    self.embedding = self.embedder(self.G, self.embedder.node_feat())

        #elif self.embModel == "Rahmen":
        #    self.embedder = Rahmen.Rahmen(self.G.edges(), in_feat, h_feat, out_feat, n_layers)
        #    self.embedding = self.embedder(self.G, self.embedder.node_feat())

        else:
            raise Exception("Unsupported embedding Model configured.")

        """
        Again, the NetWalk model is a bit different and uses its own graph
        """
        if self.embModel != "NetWalk":
            self.embedding = self.embedder(self.G)
        else:
            self.embedding = self.embedder(self.embedder.node_feat())

    """
    Training loop applying gradient descent
    """
    def train(self):
        y_true = torch.from_numpy(self.train_data.loc[:,'label'].to_numpy()).float()
        for epoch in range(self.num_epochs):
            X = torch.from_numpy(self.edge_encoder(self.train_data)).float()
            y_pred = self.pred(X)
            self.optimizer.zero_grad()
            lossfunc = nn.BCELoss()
            loss = lossfunc(y_pred, torch.unsqueeze(y_true, 1))
            loss.backward()
            self.optimizer.step()
            print('Epoch {}, loss {:.4f}'.format(epoch + 1, loss.item()))

    """
    Classifier model forward call
    """
    def forward(self, test_data):
        G, test_data = initialize_dgl(test_data)
        X = torch.from_numpy(self.edge_encoder(test_data)).float()
        return self.pred(X)

    """
    Encodes the src and dst node embeddings together as described
    in the NetWalk paper to make a combined representation.
    Also adds the edge features for classifier to include in prediction.
    """
    def edge_encoder(self, data):
        emb = self.embedding.detach().numpy()

        src = emb[data["src_id"], :]
        dst = emb[data["dst_id"], :]

        if self.encoding_method == 'Average':
            codes = (src + dst) / 2
        elif self.encoding_method == 'Hadamard':
            codes = np.multiply(src, dst)
        elif self.encoding_method  == 'WeightedL1':
            codes = abs(src - dst)
        elif self.encoding_method == 'WeightedL2':
            codes = (src - dst) ** 2

        feats = data[['pktperflow', 'byteperflow', 'tot_dur', 'flows', 'pktrate', 'tot_kbps']].to_numpy()
        combined_feats = np.concatenate((codes, feats), axis=1)
        return combined_feats
