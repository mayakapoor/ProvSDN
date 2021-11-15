import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def import_dataset():
    df = pd.read_csv("data/dataset_sdn.csv")

    df_benign = df[df['label'] == 0]
    df_attack = df[df['label'] == 1]

    train, benign_test = train_test_split(df_benign)
    test = pd.concat([df_attack, benign_test])
    test = test.sample(frac=1).reset_index(drop=True)

    return train, test

#def get_statistical_embedding(data, dimension):
