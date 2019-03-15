import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


class RelationalGraph:

    def __init__(self, network):
        self.adj_matrix = network.graph.adjacency_matrix
        self.network = network

    def create_most_similar_pairs_graph(self, n):
        from_nodes, to_nodes = self._retrieve_n_largest(n)
        from_nodes = np.apply_along_axis(self._retrieve_word, 0, from_nodes)
        to_nodes = np.apply_along_axis(self._retrieve_word, 0, to_nodes)
        self._plot_nodes(from_nodes, to_nodes, [])
        #from_list, to_list = self.create_relational_lists(from_nodes, to_nodes)
        #self._plot_nodes(from_list, to_list)

    def _retrieve_n_largest(self, n):
        x_dim, y_dim = self.adj_matrix.shape
        flat_adj_matrix = self.adj_matrix.ravel()
        top_indices = np.argsort(flat_adj_matrix)[-n:]
        coordinates = np.unravel_index(top_indices, (x_dim, y_dim))
        return coordinates

    def _retrieve_word(self, index):
        words = []
        for i in index:
            words.append(self.network.index_to_key[i])
        return np.array(words)

    @staticmethod
    def _plot_nodes(from_list, to_list, with_color_map):
        df = pd.DataFrame({'from': from_list, 'to': to_list})
        graph = nx.from_pandas_edgelist(df, 'from', 'to')
        color_map = []
        if with_color_map:
            for node in graph.nodes:
                if node in with_color_map:
                    color_map.append('green')
                else:
                    color_map.append('orange')
        else:
            color_map = ['orange']
        nx.draw(graph, with_labels=True, node_size=1000, node_color=color_map, edge_color='black', alpha=0.6, width=1, font_weight='bold')
        plt.show()

    def create_most_connected_graph(self, n):
        most_connected = self._retrieve_most_connected(n)
        from_list, to_list = self._create_relational_lists(most_connected, np.array([]))
        self._plot_nodes(from_list, to_list, [])

    def _retrieve_most_connected(self, n):
        row_totals = np.sum(self.adj_matrix, axis=0)
        indices = np.argsort(row_totals)[-n:]
        nodes = np.take(self.network.graph.nodes, indices)
        return nodes

    def _create_relational_lists(self, from_nodes, to_nodes, expand=True, sample=False):
        from_list = np.array([])
        if expand:
            from_list = from_nodes
        to_list = to_nodes
        for node in from_nodes:
            neighbors = self.network.graph.expand(node)
            if sample:
                length = len(neighbors) // 6
                neighbors = neighbors[:length]
            new_connections = np.repeat(node, len(neighbors))
            from_list = np.concatenate((from_list, new_connections))
            neighbor_words = np.array([x[0] for x in list(neighbors)])
            to_list = np.concatenate((neighbor_words, to_list))
        return from_list, to_list

    def create_custom_relational(self, words):
        from_list, to_list = self._create_relational_lists(words, np.array([]), expand=False)
        self._plot_nodes(from_list, to_list, [])

    def create_colored_path(self, path):
        #exception if empty path
        from_nodes = np.array(path[:-1])
        to_nodes = np.array(path[1:])
        from_list, to_list = self._create_relational_lists(from_nodes, to_nodes, sample=True)
        self._plot_nodes(from_list, to_list, path)


