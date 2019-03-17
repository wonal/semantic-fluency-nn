from os import getcwd        # directory operations

# Word2Vec Model Hyperparameters
W2V_ETA = 20  # TODO set to 20 when not testing

# Simulated Annealing Algorithm Settings
SA_THRESH = 0.8
SA_TEMP = 10       # starting temperature
SA_ALPHA = 0.95     # cool down alpha
SA_BETA = 1.05      # warm-up beta
WARMUP = True       # if True, reverses SA to warmup, if False, simulated annealing works as expected

# Random Start Hill Climbing Settings
HC_REPEAT = 0

# IRT Plot Settings
MAX_ITERATIONS = 200    # maximum number of iterations for algorithms during IRT comparisons
IRT_COLORS = ['#e57cb2', '#24b7f5', '#52c300']
TOTAL_TESTS = 1          # total number of times the various algorithms are run and graphs are generated

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
GRAPH_DIR = str(getcwd()) + '/images/'  # path and directory for saving graph images
TEST_DIR = str(getcwd()) + '/test_results/'       # path and directory for saving data to csv files
DELETE_GRAPHS = False                   # True clears all images, False leaves existing images

PATH_COLOR = 'green'
NODE_COLOR = 'orange'
