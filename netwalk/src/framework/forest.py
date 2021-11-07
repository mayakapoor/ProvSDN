import re
import time
import pandas as pd
import numpy as np
from detectors import LSHiForest
from datasketch import MinHash
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

def generate_codes(embedding, data):
    src = embedding[data[:, 0] - 1, :]
    dst = embedding[data[:, 1] - 1, :]

    # the edge encoding
    # refer Section 3.3 Edge Encoding in the KDD paper for details
    encoding_method = 'WeightedL1'
    if encoding_method == 'Average':
        codes = (src + dst) / 2
    elif encoding_method == 'Hadamard':
        codes = np.multiply(src, dst)
    elif encoding_method == 'WeightedL1':
        codes = abs(src - dst)
    elif encoding_method == 'WeightedL2':
        codes = (src - dst) ** 2

    return codes

def get_forest(embedding, train, k):
    codes = generate_codes(embedding, train)

    df = pd.DataFrame(codes)
    forest = LSHiForest('L1SH', k)
    #forest = IsolationForest(max_samples=100)
    start_time = time.time()
    forest.fit(df)
    train_time = time.time()-start_time
    print("\tTraining time:\t", train_time)
    return forest


def predict_forest(forest, test, labels):

    df = pd.DataFrame(test)
    start_time = time.time()
    res = forest.decision_function(df)
    test_time = time.time()-start_time
    score = 0
    ab_score = abs(np.sum(res)/(1e-10 + len(res)))
    try:
        score = roc_auc_score(labels, res)
    except ValueError:
        pass

    print("AUC for forest: " + str(score))
    print("anomaly detection score: " + str(ab_score))
    print("\tTesting time:\t", test_time)
    return score, ab_score
