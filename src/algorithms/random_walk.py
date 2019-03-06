from src.graph.Graph import UndirectedGraph
import numpy as np


class RandomWalker:
    def __init__(self, graph, start=None):
        self.g = graph
        if start:
            self.curr = start
        else:
            self.curr = None

    def run(self, count=1):
        if len(self.g.nodes) <= 0:
            raise ValueError("You must insert ")
        path = []
        if not self.curr:
            self._begin()
        for _ in range(count):
            path.append(self.curr)
            self.curr = self._next()
        return path

    def _begin(self):
        visited = set()
        c = True
        indx = np.random.randint(len(self.g.nodes))
        while self.g.adjacency_matrix[indx].sum() == 0 and len(self.g.nodes) != len(visited):
            indx = np.random.randint(len(self.g.nodes))
            visited.add(self.curr)
        self.curr = self.g._i_to_node[indx]

    def _expand(self):
        expansion = self.g.expand(self.curr)
        nodes = np.array([tup[0] for tup in expansion])
        edges = np.array([tup[1] for tup in expansion])
        prob = np.exp(edges)/np.sum(np.exp(edges))
        return nodes, edges, prob

    def _next(self):
        nodes, edges, prob = self._expand()
        if len(nodes) > 0:
            self.curr = np.random.choice(nodes, p=prob)
        return self.curr
