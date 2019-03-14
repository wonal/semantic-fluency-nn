import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


class RelationalGraph:

    def __init__(self, network):
        self.adj_matrix = network.graph.adjacency_matrix
        self.network = network

    def create_relational_graph(self, n):
        from_nodes, to_nodes = self._retrieve_n_largest(n)

        from_nodes = np.apply_along_axis(self._retrieve_word, 0, from_nodes)
        to_nodes = np.apply_along_axis(self._retrieve_word, 0, to_nodes)
        self._plot_nodes(from_nodes, to_nodes)




    def _retrieve_word(self, index):
        words = []
        for i in index:
            words.append(self.network.index_to_key[i])
        return np.array(words)

    def _retrieve_n_largest(self, n):
        x_dim, y_dim = self.adj_matrix.shape
        flat_adj_matrix = self.adj_matrix.ravel()
        top_indices = np.argsort(flat_adj_matrix)[-n:]
        coordinates = np.unravel_index(top_indices, (x_dim, y_dim))
        return coordinates

    @staticmethod
    def _plot_nodes(from_list, to_list):
        df = pd.DataFrame({'from': from_list, 'to': to_list})
        graph = nx.from_pandas_edgelist(df, 'from', 'to')
        nx.draw(graph, with_labels=True, node_size=1000, node_color="orange", edge_color='black', alpha=0.5, width=2)
        plt.show()

