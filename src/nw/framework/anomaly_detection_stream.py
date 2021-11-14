"""
    Created on: 2018-12-24
    License: BSD 3 clause

    Copyright (C) 2018
    Author: Wei Cheng <weicheng@nec-labs.com>
    Affiliation: NEC Labs America
"""
import datetime
import numpy as np
from collections import Counter
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score
from framework.forest import get_forest, predict_forest, generate_codes

def anomaly_detection_stream(embedding, train, test, k, forest):
    print('[#s] anomaly detection...\n', datetime.datetime.now())

    labels = test[:, 2]

    #calculating auc score of anomly detection task, in case that all labels are 0's or all 1's
    if np.sum(labels) == 0:
        labels[0] = 1
    elif np.sum(labels) == len(labels):
        labels[0] = 0

    auc, ab_score, accuracy = predict_forest(forest, embedding, test, labels)

    return auc, ab_score, accuracy
