from nw.framework.imports import *
import nw.framework.Model as MD
import warnings
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from nw.framework.netwalk_update import NetWalk_update

import torch
import torch.nn as nn

import numpy as np
import process

# dynamic parameters
netwalk = None                      # ref to netwalk obj
dimension = []                      # tensor size

#static parameters
activation = tf.nn.sigmoid
rho = 0.5                           # sparsity ratio
lamb = 0.0017                       # weight decay
beta = 1                            # sparsity weight
gama = 340                          # autoencoder weight
walk_len = 3                        # length of rand walks from each node
epoch = 30                          # number of epoch for optimizing, could be larger
batch_size = 40                     # should be smaller or equal to args.number_walks*n
learning_rate = 0.01                # learning rate, for adam, using 0.01, for rmsprop using 0.1
optimizer = "adam"                  #"rmsprop"#"gd"#"rmsprop" #"""gd"#""lbfgs"
corrupt_prob = [0]                  # corrupt probability, for denoising AE
ini_graph_percent = 0.01            # percent of edges in the initial graph
number_walks = 20                   # number of random walks to start at each node
snap = 100                          # number of edges in each snapshot


class NetWalk(nn.Module):
    def __init__(self, train, test, n, h, n_classes):
        super(NetWalk, self).__init__()
        self.n_nodes = n

        train_src = train['src_id']
        train_dst = train['dst_id']
        test_src = test['src_id']
        test_dst = test['dst_id']

        train_edges = []
        for src, dst in zip(train_src, train_dst):
            sublist = []
            sublist.append(src)
            sublist.append(dst)
            train_edges.append(sublist)
        train_edges = np.array(train_edges)

        test_edges = []
        for src, dst in zip(test_src, test_dst):
            sublist = []
            sublist.append(src)
            sublist.append(dst)
            test_edges.append(sublist)
        test_edges = np.array(test_edges)

        train_edges = train_edges[:, 0:2]
        test_edges = test_edges[:, 0:2]

        global dimension
        dimension = [n, h]

        data_zip = []
        data_zip.append(test_edges)
        data_zip.append(train_edges)

    # generating initial training walks
        global netwalk
        netwalk = NetWalk_update(data_zip, walk_per_node=number_walks, walk_len=walk_len,
                                init_percent=ini_graph_percent, snap=snap)

        self.model = MD.Model(activation, dimension, walk_len, n, gama, lamb, beta, rho,
                              epoch, batch_size, learning_rate, optimizer, corrupt_prob)
        self.stream = test_edges

    def node_feat(self):
        global netwalk
        ini_data = netwalk.getInitWalk()
        return ini_data

    def forward(self, walks):
        self.model.fit(walks)
        node_onehot = np.eye(self.n_nodes)
        res = self.model.feedforward_autoencoder(node_onehot)
        embedding = torch.stack([(torch.from_numpy(res[i])) for i in range(len(res))])
        return embedding

def hasNext():
    global netwalk
    return netwalk.hasNext()

def get_snapshot(df, i, snap_size):
    labels = []
    global netwalk
    for j in range(snap_size):
        labels.append(df.at[i, 'label'])
        i = i + 1
        if i == len(df):
            break
    if (netwalk.hasNext()):
        return netwalk.nextOnehotWalks(), labels
