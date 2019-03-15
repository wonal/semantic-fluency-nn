from os import getcwd        # directory operations

# Testing
TEST = False        # False means testing settings are off, True means testing settings are on
REDUCED_SIZE = 100  # reduced number of embeddings that are used to generate graphs

# Plot Parameters
RESTRICT_3D_TEXT = 300  # Restrict the number of annotations in 3D plot (computationally intensive)
TOP_N = 100             # Total similar words grabbed for each key in the cluster graph

# File/Directory Settings
GRAPH_DIR = str(getcwd()) + '/graphs/'  # path and directory for saving graph images
DELETE_GRAPHS = False                   # True clears all graphs, False leaves existing graphs

# Keys for cluster plot
CLUSTER_KEYS = ['cat', 'sun', 'virtue', 'happy', 'sleeping', 'apple',
                'poison', 'monster', 'wine', 'game', 'bird', 'heart']

# Cluster and 2D Embedding Hyperparameters
C_PERPLEXITY = 5
C_COMPONENTS = 2
C_ITER = 2400
C_ETA = 200

# 3D Embedding Hyperparameters
DIM3_PERPLEXITY = 10
DIM3_COMPONENTS = 3
DIM3_ITER = 2400
DIM3_ETA = 200
