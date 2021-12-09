import pandas as pd
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def import_dataset():
    """
    Transforms data into PD frames and normalizes NaN values
    """
    df = pd.read_csv("data/dataset_sdn.csv")

    #normalization of cols
    min_max_scaler = MinMaxScaler()
    df[["flows", "tot_dur", "pktperflow", "byteperflow", "pktrate", "tot_kbps"]] = min_max_scaler.fit_transform(df[["flows", "tot_dur", "pktperflow", "byteperflow", "pktrate", "tot_kbps"]])
    df = df.fillna(0)
    #index nodes
    uniq = set()
    for src, dst in zip(df['src'], df['dst']):
        uniq.add(src)
        uniq.add(dst)

    node_list = list(uniq)
    src_node_list = []
    dst_node_list = []

    for src, dst in zip(df['src'], df['dst']):
        src_node_list.append(node_list.index(src))
        dst_node_list.append(node_list.index(dst))

    df['src_id'] = src_node_list
    df['dst_id'] = dst_node_list

    df_benign = df[df['label'] == 0]
    df_attack = df[df['label'] == 1]

    train, benign_test = train_test_split(df_benign)
    train = train.sort_values(by='dt').reset_index(drop=True)
    test = pd.concat([df_attack, benign_test])
    test = test.sort_values(by='dt').reset_index(drop=True)

    print("Number of benign training edges: " + str(len(train)))
    print("Number of attack testing edges: " + str(len(df_attack)))
    print("Number of benign testing edges: " + str(len(benign_test)))

    print(train)
    print(test)

    return train, test, len(uniq)

def make_networkx():
    """
    Make NetworkX graph from SDN data (no features)
    """
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
    """
    Make text list representation of the graph, used by NetWalk
    """
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

def embed_embeddings(df, embedding):
    """
    Supports putting the embedding into the test or train data frame
    """
    src_embedding = []
    dst_embedding = []
    for src, dst in zip(df["src_id"], df["dst_id"]):
        src_embedding.append(embedding[src])
        dst_embedding.append(embedding[dst])
    df["src_embedding"] = src_embedding
    df["dst_embedding"] = dst_embedding
    print(df)
    return df

def edge_feat(data):
    feats = data[['pktperflow', 'byteperflow', 'tot_dur', 'flows', 'pktrate', 'tot_kbps']]
    return torch.tensor(feats.values).float()
