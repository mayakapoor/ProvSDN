from sklearn.ensemble import IsolationForest

def initialize_forest(train):
    clf = IsolationForest(n_estimators=100)
    clf.fit(train)
    return clf

def
