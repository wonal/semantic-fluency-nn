import numpy as np
import math

ALPHA = 0.95
THRESH = 0.8


class SimulatedAnnealer:

    def __init__(self, graph, initial_temp=1000, start=None):
        if initial_temp < 1:
            raise ValueError('Temperature must be greater than zero.')
        if start:
            self.start = start
        self.start = None
        self._graph = graph
        self._temperature = initial_temp
        self._initial_t = initial_temp

    def run(self, count=1):
        if len(self._graph.nodes) < 1:
            raise ValueError('You must insert.')
        if not self.start:
            self.start = self._select_random_node()
        current = (self.start, THRESH)
        return self._begin(current, count)

    def _select_random_node(self):
        visited = set()
        node = np.random.choice(self._graph.nodes)
        while not self._graph.expand(node) and len(visited) != len(self._graph.nodes):
            node = np.random.choice(self._graph.nodes)
            visited.add(node)
        return node

    def _begin(self, current_state, n):
        path = [current_state[0]]
        for i in range(1, n):
            neighbors = self._graph.expand(current_state[0])
            if not neighbors and self._temperature <= 0:
                break
            next_state = neighbors[np.random.choice(len(neighbors))]
            delta_state = next_state[1] - current_state[1]
            if delta_state < 0:
                next_state = self._select_with_p(delta_state, next_state, current_state)
            current_state = self._update(next_state, path)
            self._temperature = self._cooldown(i)
        return path

    def _select_with_p(self, delta_state, successor, current):
        e_value = math.exp(delta_state/self._temperature)
        nodes = [successor, current]
        return nodes[np.random.choice(len(nodes), p=[e_value, 1-e_value])]

    def _cooldown(self, i):
        return self._initial_t*(ALPHA**i)

    @staticmethod
    def _update(new_current, path):
        path.append(new_current[0])
        return new_current


