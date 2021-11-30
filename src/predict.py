from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import fbeta_score

import nw.model as NetWalk
import gcn.model as GCN
import sage.model as SAGE

import process
import model
import torch
import numpy as np

snap_size = 100               # number of edges in each snapshot
n_hidden  = 10                # number of hidden representations
n_out     = 2                 # number of classes
encoding_method = 'Hadamard'  # refer Section 3.3 Edge Encoding in the Netwalk KDD paper
embedder = 'GCN'              # embedding models: NetWalk, GCN, SAGE, or Spectral
optimizer = 'adam'   
learning_rate = 0.1
classifier = None             # classifiers:
n_layers = 2                  # 1D or 2D GNN


train, test, n = process.import_dataset()
mod = model.Model(n, n_hidden, n_out, embedder, encoding_method, n_layers, optimizer, learning_rate, train, test)
mod.train()

snapshotNum = 0
snapshot = test.iloc[:snap_size]
test = test.iloc[snap_size:]

while len(snapshot) == snap_size:
    snapshot = test.iloc[:snap_size]
    test = test.iloc[snap_size:]
    snapshotNum = snapshotNum + 1



# Encode the src/dst embeddings together using Hadamard
#NW_codes = helper.edge_encoder(train, NWembedding, encoding_method)
#SAGE_codes = helper.edge_encoder(train, SAGEembedding, encoding_method)
#GCN_codes = helper.edge_encoder(train, GCNembedding, encoding_method)

# cut x number of edges from testing.
# update the representation of the nodes (?)
# feed edge features plus embedding to adaptive isolation forest.


#Next, do the classification for each one. Need to reformat NW to work like SAGE/GCN. Try to
# get SAGE/GCN working first though so we know what we are doing.
