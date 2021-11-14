"""
    Created on: 2018-12-24
    License: BSD 3 clause

    Copyright (C) 2018
    Author: Wei Cheng <weicheng@nec-labs.com>
    Affiliation: NEC Labs America
"""
import datetime
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cluster import SpectralClustering


def divide_benign(ini_graph_percent, data, n, m):
    np.random.seed(1)
    print('[#s] dividing benign data into training and test...\n', datetime.datetime.now())
    train_num = int(np.floor(ini_graph_percent * m))

    # select part of edges as in the training set
    train = data[0:train_num, :]

    # select the other edges as the testing set
    test = data[train_num:, :]

    #data to adjacency_matrix
    adjacency_matrix = edgeList2Adj(data)

    # clustering nodes to clusters using spectral clustering
    kk = 3#10#42#42
    sc = SpectralClustering(kk, affinity='precomputed', n_init=100, assign_labels = 'discretize')
    labels = sc.fit_predict(adjacency_matrix)

    train_mat = csr_matrix((np.ones([np.size(train, 0)], dtype=np.int32), (train[:, 0] - 1, train[:, 1] - 1)),
                           shape=(n, n))
    train_mat = train_mat + train_mat.transpose()
    return test, train_mat, train

def create_test_set(attack, benign, n, m):
    labeled_attack = np.hstack((attack, np.ones((attack.shape[0], 1), dtype=attack.dtype)))
    labeled_benign = np.hstack((benign, np.zeros((benign.shape[0], 1), dtype=benign.dtype)))
    test = np.concatenate([labeled_attack, labeled_benign])
    np.random.shuffle(test)
    return test

def edgeList2Adj(data):
    """
    converting edge list to graph adjacency matrix
    :param data: edge list
    :return: adjacency matrix which is symmetric
    """

    data = tuple(map(tuple, data))

    n = max(max(user, item) for user, item in data)  # Get size of matrix
    matrix = np.zeros((n, n))
    for user, item in data:
        matrix[user - 1][item - 1] = 1  # Convert to 0-based index.
        matrix[item - 1][user - 1] = 1  # Convert to 0-based index.
    return matrix
