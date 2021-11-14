from skmultiflow.meta import AdaptiveRandomForestRegressor

import netwalk as nw
import numpy as np
import process

train, test = process.import_dataset()

nw.generate_netwalk_input(train, "output/train.txt")
nw.generate_netwalk_input(test, "output/test.txt")

train, stream, ini_data = nw.initialize_netwalk("output/train.txt", "output/test.txt")
embModel = nw.get_model()
#forest = isof.initialize_forest(train)

embedding = nw.get_embedding(embModel, ini_data)

snapshotNum = 1
while(nw.hasNext()):
    snapshot = nw.get_snapshot(stream)
    embedding = nw.get_embedding(embModel, snapshot)
