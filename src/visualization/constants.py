from os import getcwd        # directory operations

# Testing
TEST = False        # False means testing settings are off, True means testing settings are on
TEST_SIZE = 100     # total number of embeddings that are used to generate graphs

# Plot Parameters
RESTRICT_3D_TEXT = 700  # Restrict the number of annotations in 3D plot (computationally intensive)
TOP_N = 100             # Total similar words grabbed for each key in the cluster graph

# File/Directory Settings
GRAPH_DIR = str(getcwd()) + '/graphs/'  # path and directory for saving graph images
DELETE_GRAPHS = False                   # True clears all graphs, False leaves existing graphs