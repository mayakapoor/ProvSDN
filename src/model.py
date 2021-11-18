from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import fbeta_score

from skmultiflow.meta import AdaptiveRandomForestRegressor
from skmultiflow.trees import HoeffdingTreeClassifier

from math import sqrt

import netwalk as nw
import numpy as np
import process

snap = 100                          # number of edges in each snapshot
k = 100                             # number of forests

train, test = process.import_dataset()

nw.generate_netwalk_input(train, "output/train.txt")
nw.generate_netwalk_input(test, "output/test.txt")

stream, ini_data = nw.initialize_netwalk("output/train.txt", "output/test.txt")
embModel = nw.get_model()
#forest = isof.initialize_forest(train)
print(train)
embedding = nw.get_embedding(embModel, ini_data)
codes = nw.generate_codes(embedding, ini_data)
print(codes)

ht = HoeffdingTreeClassifier()
y_true = np.zeros(len(codes))
ht = ht.partial_fit(codes, y_true)

y_pred = np.zeros(len(test))
y_true = np.zeros(len(test))

snapshotNum = 0
correct_cnt = 0

while(nw.hasNext()):
    snapshot, labels = nw.get_snapshot(test, snapshotNum*snap, snap, len(test))
    embedding = nw.get_embedding(embModel, snapshot)
    print(snapshot)
    codes = nw.generate_codes(embedding, snapshot)
    y_pred = ht.predict(snapshot)

    for j in range(len(snapshot)):
        if labels[j] == y_pred[j]:
            correct_cnt = correct_cnt + 1

    ht.partial_fit(codes, labels)
    snapshotNum = snapshotNum + 1

# Display results
print('{} samples analyzed.'.format(snapshotNum))
print('Hoeffding Tree accuracy: {}'.format(correct_cnt / len(test)))
