from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import fbeta_score
from skmultiflow.trees import HoeffdingTreeClassifier

from math import sqrt

import netwalk as nw
import numpy as np
import process

from forest import generate_codes, generate_forest, predict_forest

snap = 100                          # number of edges in each snapshot
k = 100                             # number of forests

train, test = process.import_dataset()

nw.generate_netwalk_input(train, "output/train.txt")
nw.generate_netwalk_input(test, "output/test.txt")

train, stream, ini_data = nw.initialize_netwalk("output/train.txt", "output/test.txt")
embModel = nw.get_model()
#forest = isof.initialize_forest(train)

embedding = nw.get_embedding(embModel, ini_data)
codes = generate_codes(embedding, ini_data)

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
    codes = generate_codes(embedding, snapshot)
    #y_true[snapshotNum] = labels[0]
    #y_pred[snapshotNum] = arf_reg.predict(codes)[0]
    y_pred = ht.predict(codes)
    if labels[0] == y_pred[0]:
        correct_cnt = correct_cnt + 1
    ht.partial_fit(codes, labels)
    #arf_reg.partial_fit(embedding, labels)
    snapshotNum = snapshotNum + 1

# Display results
print('{} samples analyzed.'.format(snapshotNum))
print('Hoeffding Tree accuracy: {}'.format(correct_cnt / snapshotNum))
