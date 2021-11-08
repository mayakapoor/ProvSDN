import csv
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

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

train = open("output/benign.txt", "w")
prev_entry = ""
for row in my_data:
    if (row[22] == 0):
        entry = str(row[2]) + " " + str(row[3]) + " " + str(row[4]) + " " + str(row[0]) + "\n";
        if (entry != prev_entry):
            train.write(entry)
        prev_entry = entry


print(graph_count)
print(len(benign_graphs))
print(len(attack_graphs))
#nx.draw(benign_graphs[200][1], with_labels=True, cmap = plt.get_cmap('jet'))
#plt.show()
