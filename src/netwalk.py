from nw.framework.imports import *
import nw.framework.Model as MD
import warnings
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from nw.framework.netwalk_update import NetWalk_update


# dynamic parameters
n = 0                               # number of nodes
netwalk = None                      # ref to netwalk obj

#static parameters
hidden_size = 19                   # number of latent dimensions to learn
activation = tf.nn.sigmoid
dimension = [n, hidden_size]
rho = 0.5                           # sparsity ratio
lamb = 0.0017                       # weight decay
beta = 1                            # sparsity weight
gama = 340                          # autoencoder weight
walk_len = 3                        # length of rand walks from each node
epoch = 30                          # number of epoch for optimizing, could be larger
batch_size = 40                     # should be smaller or equal to args.number_walks*n
learning_rate = 0.01                # learning rate, for adam, using 0.01, for rmsprop using 0.1
optimizer = "adam"                  #"rmsprop"#"gd"#"rmsprop" #"""gd"#""lbfgs"
corrupt_prob = [0]                  # corrupt probability, for denoising AE
ini_graph_percent = 0.01            # percent of edges in the initial graph
number_walks = 20                   # number of random walks to start at each node
snap = 100                          # number of edges in each snapshot
k = 100                             # number of trees to build

def remove_self_loops(edges):
    idx_remove_dups = np.nonzero(edges[:, 0] - edges[:, 1] < 0)
    edges = edges[idx_remove_dups]

    edges = edges[:, 0:2]

    step = int(np.floor(1/1))
    edges = edges[0:len(edges):step, :]
    np.random.seed(101)
    np.random.shuffle(edges)

def initialize_netwalk(train_path, test_path):
    train_edges = np.loadtxt(train_path, dtype=int, comments='%') + 1
    test_edges = np.loadtxt(test_path, dtype=int, comments='%') + 1

    uniq = set()
    for edge in train_edges:
        for node in edge:
            uniq.add(node)
    for edge in test_edges:
        for node in edge:
            uniq.add(node)

    global n
    global dimension
    global hidden_size
    n = len(uniq)
    dimension = [n, hidden_size]

    remove_self_loops(train_edges)
    remove_self_loops(test_edges)

    data_zip = []
    data_zip.append(test_edges)
    data_zip.append(train_edges)

# generating initial training walks
    global netwalk
    netwalk = NetWalk_update(data_zip, walk_per_node=number_walks, walk_len=walk_len,
                            init_percent=ini_graph_percent, snap=snap)

# learning initial embeddings for training edges
    ini_data = netwalk.getInitWalk()
    return train_edges, test_edges, ini_data

def hasNext():
    global netwalk
    return netwalk.hasNext()

def generate_netwalk_input(data, path):
    """""
    sort edges by timestamp and write src, dest nodes to a text file.
    """""

    data_file = open(path, "w")
    data.sort_values('dt')

    nodes = np.concatenate([data['src'], data['dst']])
    unodes = np.unique(nodes)
    print(unodes)
    nodelist = unodes.tolist()

    for src, dst in zip(data['src'], data['dst']):
        entry = str(nodelist.index(src)) + " " + str(nodelist.index(dst)) + "\n"
        data_file.write(entry)

def get_model():
    embModel = MD.Model(activation, dimension, walk_len, n, gama, lamb, beta, rho,
                        epoch, batch_size, learning_rate, optimizer, corrupt_prob)
    return embModel

def get_embedding(model, data):
    """
        function getEmbedding(model, data, n)
        #  the function feed ''data'' which is a list of walks
        #  the embedding ''model'', n: the number of total nodes
        return: the embeddings of all nodes, each row is one node, each column is one dimension of embedding
    """
    # batch optimizing to fit the model
    model.fit(data)

    # Retrieve the embeddings
    node_onehot = np.eye(n)
    res = model.feedforward_autoencoder(node_onehot)
    return res

def get_snapshot(data):
    if (netwalk.hasNext()):
        return netwalk.nextOnehotWalks()
