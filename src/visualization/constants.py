from os import getcwd        # directory operations

# Plot Parameters
RESTRICT_3D_TEXT = 150  # Restrict the number of annotations in 3D plot (computationally intensive)
TOP_N = 100             # Total similar words grabbed for each key in the cluster graph

# File/Directory Settings
GRAPH_DIR = str(getcwd()) + '/images/'  # path and directory for saving graph images
DELETE_GRAPHS = False                         # True clears all images, False leaves existing images
IMAGE_DIRECTORY = str(getcwd()) + '/images/'

# Corpus name and keys for cluster plot
CLUSTER_KEYS = ['cat', 'sun', 'virtue', 'happy', 'sleeping', 'apple',
                'poison', 'monster', 'wine', 'game', 'bird', 'heart']

# Word2Vec Model Hyperparemeters
W2V_ETA = 20

# Cluster and 2D Embedding Hyperparameters
C_PERPLEXITY = 5
C_COMPONENTS = 2
C_ITER = 2400
C_ETA = 350

# 3D Embedding Hyperparameters
DIM3_PERPLEXITY = 10
DIM3_COMPONENTS = 3
DIM3_ITER = 2400
DIM3_ETA = 500
