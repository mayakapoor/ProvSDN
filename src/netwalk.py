from nw.framework.imports import *
import nw.framework.Model as MD
import warnings
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from nw.framework.netwalk_update import NetWalk_update


# dynamic parameters
netwalk = None                      # ref to netwalk obj
dimension = []                      # tensor size

#static parameters
hidden_size = 20                   # number of latent dimensions to learn
activation = tf.nn.sigmoid

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

def initialize_netwalk(train_src, train_dst, test_src, test_dst, n):

    train_edges = []
    for src, dst in zip(train_src, train_dst):
        sublist = []
        sublist.append(src)
        sublist.append(dst)
        train_edges.append(sublist)
    train_edges = np.array(train_edges)

    test_edges = []
    for src, dst in zip(test_src, test_dst):
        sublist = []
        sublist.append(src)
        sublist.append(dst)
        test_edges.append(sublist)
    test_edges = np.array(test_edges)

    train_edges = train_edges[:, 0:2]
    test_edges = test_edges[:, 0:2]

    global dimension
    global hidden_size
    dimension = [n, hidden_size]

    #remove_self_loops(train_edges)
    #remove_self_loops(test_edges)

    data_zip = []
    data_zip.append(test_edges)
    data_zip.append(train_edges)

# generating initial training walks
    global netwalk
    netwalk = NetWalk_update(data_zip, walk_per_node=number_walks, walk_len=walk_len,
                            init_percent=ini_graph_percent, snap=snap)

# learning initial embeddings for training edges
    ini_data = netwalk.getInitWalk()
    return test_edges, ini_data

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
    nodelist = unodes.tolist()

    for src, dst in zip(data['src'], data['dst']):
        entry = str(nodelist.index(src)) + " " + str(nodelist.index(dst)) + "\n"
        data_file.write(entry)

def get_model(n):
    embModel = MD.Model(activation, dimension, walk_len, n, gama, lamb, beta, rho,
                        epoch, batch_size, learning_rate, optimizer, corrupt_prob)
    return embModel

def get_embedding(model, data, n):
    """
        function getEmbedding(model, data, n)
        #  the function feed ''data'' which is a list of walks
        #  the embedding ''model'', n: the number of total nodes
        return: the embeddings of all nodes, each row is one node, each column is one dimension of embedding
    """
    # batch optimizing to fit the model
    model.fit(data)
    global netwalk

    # Retrieve the embeddings
    node_onehot = np.eye(n)
    res = model.feedforward_autoencoder(node_onehot)

    #save the embeddings into a reference in netwalk
    for i in range(len(netwalk.vertices)):
        netwalk.vertices[i] = res[i]
    return netwalk.vertices

def get_snapshot(df, i, snap_size, stop):
    labels = []
    for j in range(snap_size):
        labels.append(df.at[i, 'label'])
        i = i + 1
        if i == stop:
            break
    if (netwalk.hasNext()):
        return netwalk.nextOnehotWalks(), labels
