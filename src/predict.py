from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

import process
import model
import torch
import numpy as np

snap_size = 100               # number of edges in each snapshot
n_hidden  = 15                # number of hidden representations
n_out     = 10                # number of output features
encoding_method = 'Hadamard'  # refer Section 3.3 Edge Encoding in the Netwalk KDD paper
embedder = 'SAGE'             # embedding models: NetWalk, GCN, SAGE, Rahmen, or Spectral
optimizer = 'adam'            # optimizers:
learning_rate = 0.01          # learning rate
n_layers = 2                  # number of layers for network
n_epochs = 20                 # number of training epochs

train, test, n = process.import_dataset()
mod = model.Model(n, n_hidden, n_out, embedder, encoding_method, n_layers, optimizer, learning_rate, n_epochs, train, test)
mod.train()


snapshotNum = 0
snapshot = test.iloc[:snap_size]
test = test.iloc[snap_size:]
total = 0
y_pred = []
y_true = []

with torch.no_grad():
    while len(snapshot) == snap_size:
        snapshot = test.iloc[:snap_size]
        test = test.iloc[snap_size:]
        total += snap_size
        for item in mod(snapshot).detach().numpy():
            y_pred.append(item)
        for item in snapshot['label'].tolist():
            y_true.append(item)
        snapshotNum = snapshotNum + 1
num_true = int(sum(y_true))
sorted_pred = sorted(y_pred)
threshold = sorted_pred[-num_true]

y_pred = [
    1 if pred > threshold else 0
    for pred in y_pred
]

print("Accuracy: %.2f%%" % ((accuracy_score(y_true, y_pred, normalize=False)) / total * 100))
print("AUC-ROC: " + str(roc_auc_score(y_true, y_pred)))
print("Precision: %.2f%%" % (precision_score(y_true, y_pred) * 100))
print("Recall: %.2f%%" % (recall_score(y_true, y_pred) * 100))
print("F1-Score: %.2f%%" % (f1_score(y_true, y_pred) * 100))
