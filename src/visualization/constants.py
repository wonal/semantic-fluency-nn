from os import getcwd         # directory operations

# Word2Vec Model Hyperparameters
W2V_ETA = 20

# Simulated Annealing Algorithm Settings
SA_ALPHA = 0.95
SA_THRESH = 0.8
SA_TEMP = 3

# Random Start Hill Climbing Settings
HC_REPEAT = 0

# IRT Plot Settings
TOTAL_TESTS = 1          # total number of times the various algorithms are run and graphs are generated
MAX_ITERATIONS = 200   # maximum number of iterations for algorithms during IRT comparisons
IRT_COLORS = ['#e57cb2', '#24b7f5', '#52c300']

# Networkx Plot Coloring
PATH_COLOR = 'green'
NODE_COLOR = 'orange'

# TSNE Plot Settings
RESTRICT_3D_TEXT = 125   # Restrict the number of annotations in 3D plot (computationally intensive)
TOP_N = 100             # Total similar words grabbed for each key in the cluster graph

# Corpus name and keys for cluster plot
CLUSTER_KEYS = ['cat', 'sun', 'virtue', 'happy', 'sleeping', 'apple',
                'poison', 'monster', 'wine', 'game', 'bird', 'heart']

# Cluster and 2D Embedding Hyperparameters
C_PERPLEXITY = 5
C_COMPONENTS = 2
C_ITER = 2400
C_ETA = 450

# 3D Embedding Hyperparameters
DIM3_PERPLEXITY = 10
DIM3_COMPONENTS = 3
DIM3_ITER = 2400
DIM3_ETA = 400

# File/Directory Settings
GRAPH_DIR = str(getcwd()) + '/data/output/images/'       # path and directory for saving graph images
TEST_DIR = str(getcwd()) + '/data/output/test_results/'       # path and directory for saving data to csv files
DELETE_GRAPHS = False                   # True clears all images, False leaves existing images


