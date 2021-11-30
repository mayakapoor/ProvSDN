import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def import_dataset():
    df = pd.read_csv("data/dataset_sdn.csv")

    #normalization of cols
    min = df.pktcount.min()
    max = df.pktcount.max()
    df['pktcount'] = (df.pktcount-min)/(max-min)

    min = df.bytecount.min()
    max = df.bytecount.max()
    df['bytecount'] = (df.bytecount-min)/(max-min)

    min = df.dur.min()
    max = df.dur.max()
    df['dur'] = (df.dur-min)/(max-min)

    min = df.packetins.min()
    max = df.packetins.max()
    df['packetins'] = (df.packetins-min)/(max-min)

    min = df.pktperflow.min()
    max = df.pktperflow.max()
    df['pktperflow'] = (df.pktperflow-min)/(max-min)

    min = df.byteperflow.min()
    max = df.byteperflow.max()
    df['byteperflow'] = (df.byteperflow-min)/(max-min)

    min = df.pktrate.min()
    max = df.pktrate.max()
    df['pktrate'] = (df.pktrate-min)/(max-min)

    min = df.pktcount.min()
    max = df.pktcount.max()
    df['pktcount'] = (df.pktcount-min)/(max-min)

    min = df.tx_bytes.min()
    max = df.tx_bytes.max()
    df['tx_bytes'] = (df.tx_bytes-min)/(max-min)

    min = df.rx_bytes.min()
    max = df.rx_bytes.max()
    df['rx_bytes'] = (df.rx_bytes-min)/(max-min)

    min = df.tx_kbps.min()
    max = df.tx_kbps.max()
    df['tx_kbps'] = (df.tx_kbps-min)/(max-min)

    min = df.rx_kbps.min()
    max = df.rx_kbps.max()
    df['rx_kbps'] = (df.rx_kbps-min)/(max-min)

    min = df.tot_kbps.min()
    max = df.tot_kbps.max()
    df['tot_kbps'] = (df.tot_kbps-min)/(max-min)

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

def embed_embeddings(df, embedding):
    src_embedding = []
    dst_embedding = []
    for src, dst in zip(df["src_id"], df["dst_id"]):
        src_embedding.append(embedding[src])
        dst_embedding.append(embedding[dst])
    df["src_embedding"] = src_embedding
    df["dst_embedding"] = dst_embedding
    print(df)
    return df

def extract_edge_features(data):
    return data[['pktcount', 'bytecount', 'dur', 'packetins', 'pktperflow', 'byteperflow', 'pktrate', 'pktcount', 'tx_bytes', 'rx_bytes', 'tx_kbps', 'rx_kbps', 'tot_kbps']]

def get_snapshot_labels(df, i, snap_size):
    labels = []
    global netwalk
    for j in range(snap_size):
        labels.append(df.at[i, 'label'])
        i = i + 1
        if i == len(df):
            break
    return labels
