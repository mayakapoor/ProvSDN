"""
    Created on: 2018-12-24
    License: BSD 3 clause

    Copyright (C) 2018
    Author: Wei Cheng <weicheng@nec-labs.com>
    Affiliation: NEC Labs America
"""
import datetime
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score
from framework.forest import get_forest, predict_forest, generate_codes

def anomaly_detection_stream(embedding, train, synthetic_test, k, alfa, forest):
    """
    function anomaly_detection_stream(embedding, train, synthetic_test, k, alfa, n0, c0)
    #  the function generate codes of edges by combining embeddings of two
    #  nodes, and then using the testing codes of edges for anomaly detection
    #  Input: embedding: embeddings of each node; train: training edges; synthetic_test: testing edges with anomlies;
                k: number of clusters; alfa: updating rate; n0: last round number of nodes in each cluster;
                c0: cluster centroids in last round
    #  return scores: The anomaly severity ranking, the top-ranked are the most likely anomlies
    #   auc: AUC score
    #   n:   number of nodes in each cluster
    #   c:   cluster centroids,
    #   res: id of nodes if their distance to nearest centroid is larger than that in the training set
    #   ab_score: anomaly score for the whole snapshot, just the sum of distances to their nearest centroids
    """

    print('[#s] edge encoding...\n', datetime.datetime.now())

    codes = generate_codes(embedding, synthetic_test)

    print('[#s] anomaly detection...\n', datetime.datetime.now())

    labels = synthetic_test[:, 2]

    #calculating auc score of anomly detection task, in case that all labels are 0's or all 1's
    if np.sum(labels) == 0:
        labels[0] = 1
    elif np.sum(labels) == len(labels):
        labels[0] = 0

    auc, ab_score = predict_forest(forest, codes, labels)

    return auc, ab_score