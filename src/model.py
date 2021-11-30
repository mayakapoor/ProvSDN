import nw.model as NetWalk
import gcn.model as GCN
import sage.model as SAGE

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def initialize_dgl(train):
    edges = []
    for src, dst in zip(train["src_id"], train["dst_id"]):
        edges.append((src, dst))
    G = dgl.graph(edges)

    G.edata['label'] = torch.from_numpy(train[['label']].to_numpy())
    G.edata['pktcount'] = torch.from_numpy(train[['pktcount']].to_numpy())
    G.edata['bytecount'] = torch.from_numpy(train[['bytecount']].to_numpy())
    G.edata['dur'] = torch.from_numpy(train[['dur']].to_numpy())
    G.edata['packetins'] = torch.from_numpy(train[['packetins']].to_numpy())
    G.edata['pktperflow'] = torch.from_numpy(train[['pktperflow']].to_numpy())
    G.edata['byteperflow'] = torch.from_numpy(train[['byteperflow']].to_numpy())
    G.edata['pktrate'] = torch.from_numpy(train[['pktrate']].to_numpy())
    G.edata['tx_bytes'] = torch.from_numpy(train[['tx_bytes']].to_numpy())
    G.edata['rx_bytes'] = torch.from_numpy(train[['rx_bytes']].to_numpy())
    G.edata['tx_kbps'] = torch.from_numpy(train[['tx_bytes']].to_numpy())
    G.edata['rx_kbps'] = torch.from_numpy(train[['rx_bytes']].to_numpy())
    G.edata['tot_kbps'] = torch.from_numpy(train[['tot_kbps']].to_numpy())

    G = dgl.add_self_loop(G)
    return G

class Model(nn.Module):
    def __init__(self, in_feat, h_feat, out_feat, embModel, encoding_method,
                 n_layers, optimizer_type, learning_rate, train, test):
        super(Model, self).__init__()
        self.in_feat = in_feat
        self.h_feat = h_feat
        self.n_classes = out_feat
        self.embModel = embModel
        self.encoding_method = encoding_method
        self.n_layers = n_layers
        self.train_data = train
        self.test_data = test
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.G = initialize_dgl(self.train_data)
        self.pred = nn.Linear(self.h_feat, self.n_classes)

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

        if self.embModel == "SAGE":
            self.embedder = SAGE.SAGE(in_feat, h_feat, out_feat, n_layers)
            self.embedding = self.embedder.forward(self.G, self.embedder.node_feat())

        # netwalk uses its own graph structure, not DGL
        # also trains with feedforward autoencoder
        elif self.embModel == "NetWalk":
            self.embedder = NetWalk.NetWalk(train, test, in_feat, h_feat, out_feat)
            self.embedding = self.embedder.forward(self.embedder.node_feat())

        elif self.embModel == "GCN":
            self.embedder = GCN.GCN(in_feat, h_feat, out_feat, n_layers)
            self.embedding = self.embedder.forward(self.G, self.embedder.node_feat())

        #elif self.embModel == "Spectral":

        #elif self.embModel == "Rahmen":

        else:
            raise Exception("Unsupported embedding Model configured.")

    def edge_encoder(self, data):
        emb = self.embedding.detach().numpy()
        edges = self.G.edges()
        src = emb[data["src_id"], :]
        dst = emb[data["dst_id"], :]

        # the edge encoding
        # refer Section 3.3 Edge Encoding in the KDD paper for details
        if self.encoding_method == 'Average':
            codes = (src + dst) / 2
        elif self.encoding_method == 'Hadamard':
            codes = np.multiply(src, dst)
        elif self.encoding_method == 'WeightedL1':
            codes = abs(src - dst)
        elif self.encoding_method == 'WeightedL2':
            codes = (src - dst) ** 2
        return codes

    def train(self):
        edge_label = torch.from_numpy(self.train_data.loc[:,'label'].to_numpy())
        for epoch in range(10):
            pred = self.pred(torch.from_numpy(self.edge_encoder(self.train_data)))
            loss = F.cross_entropy(pred, edge_label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print(loss.item())
