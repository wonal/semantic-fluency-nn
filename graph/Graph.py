import numpy as np


class UndirectedGraph:
    def __init__(self, initial_dim=2, mult_factor=2):
        if mult_factor < 2:
            raise ValueError('Multiplication factor must be 2 or more.')
        if initial_dim < 1:
            raise ValueError('Initial dimensions must be 1 or more')
        self._nodes = {}
        self._curr = 0
        self._dim = initial_dim
        self._mult_factor = mult_factor
        self.adjacency_matrix = np.zeros((self._dim, self._dim))

    def add(self, node_1, node_2, edge):
        self._add_node(node_1)
        self._add_node(node_2)
        self._add_edge(node_1, node_2, edge)

    def get_edge(self, node_1, node_2):
        i = self._nodes[node_1]
        j = self._nodes[node_2]
        edge = self.adjacency_matrix[i][j]
        return edge

    def expand(self, node, sort=False, reverse=False):
        vector = self.adjacency_matrix[self._nodes[node]]
        expansion = []
        for i in range(0, len(vector)):
            if vector[i] != 0:
                expansion.append((self.nodes[i], vector[i]))
        if sort:
            expansion = sorted(expansion, key=lambda tup: tup[1])
        if reverse:
            expansion = list(reversed(expansion))
        return expansion

    @property
    def nodes(self):
        return list(self._nodes.keys())

    def _add_node(self, node):
        if node not in self._nodes.keys():
            self._nodes[node] = self._curr
            self._curr += 1

        if self._curr >= self._dim:
            self._resize()

    def _resize(self):
        old = self.adjacency_matrix
        self._dim *= self._mult_factor
        self.adjacency_matrix = np.zeros((self._dim, self._dim))
        self.adjacency_matrix[:len(old), :len(old)] = old

    def _add_edge(self, node_1, node_2, edge):
        if node_1 != node_2:
            i = self._nodes[node_1]
            j = self._nodes[node_2]
            self.adjacency_matrix[i][j] = edge
            self.adjacency_matrix[j][i] = edge
