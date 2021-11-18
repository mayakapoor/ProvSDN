import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def import_dataset():
    df = pd.read_csv("data/dataset_sdn.csv")

    df_benign = df[df['label'] == 0]
    df_attack = df[df['label'] == 1]

    train, benign_test = train_test_split(df_benign)
    train = train.reset_index(drop=True)
    test = pd.concat([df_attack, benign_test])
    test = test.sample(frac=1).reset_index(drop=True)

    return train, test

def make_networkx():
    my_data = np.genfromtxt('data/dataset_sdn.csv', delimiter=',',skip_header=1,dtype=None, encoding='UTF-8')
    my_data.sort()

    graph_count = 0
    has_attack = False
    curr_dt = my_data[0][0]
    curr_graph = []
    benign_graphs = []
    attack_graphs = []

    G = nx.MultiDiGraph()

    for row in my_data:
        if row[0] != curr_dt:
            curr_dt = row[0]
            graph_count = graph_count + 1
            if has_attack:
                attack_graphs.append((curr_graph, G))
            else:
                benign_graphs.append((curr_graph, G))
            curr_graph = []
            has_attack = False
            G = nx.DiGraph()
        if row[22] == 1:
            has_attack = True
        curr_graph.append(row)
        G.add_edge(row[2], row[3])

    #don't forget the last one
    graph_count = graph_count + 1
    if has_attack:
        attack_graphs.append((curr_graph, G))
    else:
        benign_graphs.append((curr_graph, G))

    print("Total number of graphs: " + str(graph_count))
    print("Total number of benign graphs: " + str(len(benign_graphs)))
    print("Total number of attack graphs: " + str(len(attack_graphs)))

def make_txt_graph():
    print("Beginning data pre-processing...\n")

    my_data = np.genfromtxt('data/dataset_sdn.csv', delimiter=',',skip_header=1,dtype=None, encoding='UTF-8')
    my_data.sort()

    train = open("output/train.txt", "w")
    test = open("output/test.txt", "w")
    node_map = open("output/nodes.txt", "w")
    prev_entry = ""
    nodes = {}

    for row in my_data:
        if row[2] not in nodes:
            nodes[row[2]] = len(nodes)
        if row[3] not in nodes:
            nodes[row[3]] = len(nodes)

        entry = str(nodes[row[2]]) + " " + str(nodes[row[3]]) + "\n"
        if (entry != prev_entry):
            if (row[22] == 0):
                train.write(entry)
            else:
                test.write(entry)
            prev_entry = entry

    for node in nodes:
        entry = str(node) + " " + str(nodes[node]) + "\n"
        node_map.write(entry)
make_txt_graph()
#def get_statistical_embedding(data, dimension):
