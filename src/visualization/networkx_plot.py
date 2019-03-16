import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import src.visualization.constants as C


class NetworkxPlot:

    def __init__(self, network):
        self.adj_matrix = network.graph.adjacency_matrix
        self.graph = network.graph
        self.index_to_key = network.index_to_key

    def plot_most_similar_pairs(self, n, expanded=False):
        from_nodes, to_nodes = self._retrieve_n_largest(n)
        from_nodes = np.apply_along_axis(self._retrieve_word, 0, from_nodes)
        to_nodes = np.apply_along_axis(self._retrieve_word, 0, to_nodes)
        if expanded:
            from_nodes, to_nodes = self._expand_edge_lists(from_nodes, to_nodes)
        self._plot_nodes(from_nodes, to_nodes, [])

    def _retrieve_n_largest(self, n):
        x_dim, y_dim = self.adj_matrix.shape
        flat_adj_matrix = self.adj_matrix.ravel()
        top_indices = np.argsort(flat_adj_matrix)[-n:]
        coordinates = np.unravel_index(top_indices, (x_dim, y_dim))
        return coordinates

    def _retrieve_word(self, index):
        words = []
        for i in index:
            words.append(self.index_to_key[i])
        return np.array(words)

    @staticmethod
    def _plot_nodes(from_list, to_list, with_color_map):
        df = pd.DataFrame({'from': from_list, 'to': to_list})
        graph = nx.from_pandas_edgelist(df, 'from', 'to')
        color_map = []
        if with_color_map:
            for node in graph.nodes:
                if node in with_color_map:
                    color_map.append(C.PATH_COLOR)
                else:
                    color_map.append(C.NODE_COLOR)
        else:
            color_map = [C.NODE_COLOR]
        nx.draw(graph, with_labels=True, node_size=1000, node_color=color_map, edge_color='black', alpha=0.6, width=1, font_weight='bold')
        plt.show()

    def _expand_edge_lists(self, from_nodes, to_nodes, have_edge_lists=True, sample=False):
        from_list = np.array([])
        if have_edge_lists:
            from_list = from_nodes
        to_list = to_nodes
        for node in from_nodes:
            try:
                neighbors = self.graph.expand(node)
                if sample:
                    length = len(neighbors) // 6
                    neighbors = neighbors[:length]
                new_connections = np.repeat(node, len(neighbors))
                from_list = np.concatenate((from_list, new_connections))
                neighbor_words = np.array([x[0] for x in list(neighbors)])
                to_list = np.concatenate((to_list, neighbor_words))
            except KeyError:
                continue
        return from_list, to_list

    def plot_most_connected(self, n):
        if n < 1 or n > len(self.graph.nodes):
            raise ValueError("n must be greater than zero and less than the number of nodes in the graph.")
        most_connected = self._retrieve_most_connected(n)
        from_list, to_list = self._expand_edge_lists(most_connected, np.array([]), have_edge_lists=False)
        self._plot_nodes(from_list, to_list, [])

    def _retrieve_most_connected(self, n):
        row_totals = np.sum(self.adj_matrix, axis=0)
        indices = np.argsort(row_totals)[-n:]
        nodes = np.take(self.graph.nodes, indices)
        return nodes

    def create_custom_plot(self, words):
        if not words:
            raise ValueError("Must enter in at least one word.")
        from_list, to_list = self._expand_edge_lists(words, np.array([]), have_edge_lists=False)
        self._plot_nodes(from_list, to_list, [])

    def create_colored_path_plot(self, path):
        if len(path) < 2:
            raise ValueError("Path length must be at least 2")
        from_nodes = np.array(path[:-1])
        to_nodes = np.array(path[1:])
        from_list, to_list = self._expand_edge_lists(from_nodes, to_nodes, sample=True)
        self._plot_nodes(from_list, to_list, path)


