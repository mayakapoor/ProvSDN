from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import fbeta_score

from skmultiflow.meta import AdaptiveRandomForestRegressor
from skmultiflow.trees import HoeffdingTreeClassifier

from math import sqrt

import dglearn.model as dglearn

import netwalk as nw
import numpy as np
import process

snap = 100                          # number of edges in each snapshot

train, test, n = process.import_dataset()

# STEP 1: generate initial node embedding for training graph and prepare
#         the data stream for testing.
#         This prepares the walks for test edges as well.
stream, ini_data = nw.initialize_netwalk(train['src_id'], train['dst_id'], test['src_id'], test['dst_id'], n)
embModel = nw.get_model(n)
embedding = nw.get_embedding(embModel, ini_data, n)

# STEP 2: embed the embeddings into the edge data by replacing the src/dst node with
#         their learned hidden representations.
process.embed_embeddings(train, embedding)

# STEP 3: build the DGL graph and train it
G = dglearn.initialize_dgl(train)
y_pred = np.zeros(len(test))
y_true = np.zeros(len(test))

snapshotNum = 0
correct_cnt = 0



while(nw.hasNext()):
    snapshot, labels = nw.get_snapshot(test, snapshotNum*snap, snap, len(test))
    embedding = nw.get_embedding(embModel, snapshot, n)

    y_pred = arf.predict(embedding)
    arf.partial_fit(embedding, labels)

    for j in range(len(snapshot)):
        if labels[j] == y_pred[j]:
            correct_cnt = correct_cnt + 1

    snapshotNum = snapshotNum + 1

# Display results
print('{} samples analyzed.'.format(snapshotNum))
print('Adaptive Random Forest Regressor accuracy: {}'.format(correct_cnt / len(test)))
