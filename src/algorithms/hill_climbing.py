"""
Random Start Hill Climbing

Syntax: my_hill_climber = HillClimber(my_graph, start, num_ter, repeat)

Inputs:
        - my_graph, a graph object
        - start, a starting node on the graph
        - num_iter, a maximum amount of iterations
        - repeat, the maximum consecutive visitations to a single node

Function:
        - run(), begins the hill climbing. will return the list of traversed nodes.

Based on pseudocode from below (source: wikipedia.com)
Discrete Space Hill Climbing Algorithm
   currentNode = startNode;
   loop do
      L = NEIGHBORS(currentNode);
      nextEval = -INF;
      nextNode = NULL;
      for all x in L
         if (EVAL(x) > nextEval)
              nextNode = x;
              nextEval = EVAL(x);
      if nextEval <= EVAL(currentNode)
         //Return current node since no better neighbors exist
         return currentNode;
      currentNode = nextNode;
"""

from src.graph.Graph import UndirectedGraph
import numpy as np


class HillClimber:

    def __init__(self, graph, start=None, num_iter=20, repeat=3):
        self.g = graph
        if start:
            self.curr = start
        else:
            self.curr = self._random_start()
        self.score = float('-Inf')
        self.explored = set()
        self.list = []
        self.num_iter = num_iter
        self.repeat = repeat

    def run(self):

        explore = True
        cur_iter = 0
        repeat = 0

        while explore:
            self.explored.add(self.curr)
            self.list.append(self.curr)
            nodes, edges = self._expand()

            next_eval = float('-Inf')
            next_node = None
            for node in nodes:
                self.explored.add(node)
                idx, = np.where(nodes == node)
                if edges[idx] > next_eval:
                    next_node = node
                    next_eval = edges[idx]
            if cur_iter >= self.num_iter:
                return self.list
            else:
                if next_eval == self.score:
                    repeat += 1
                self.curr = next_node
                self.score = next_eval
                self.list.append(next_node)
                if repeat >= self.repeat:
                    self.curr = self._random_start()
                    repeat = 0
            cur_iter += 1

        return self.list

    def _expand(self):
        expansion = self.g.expand(self.curr)
        nodes = np.array([tup[0] for tup in expansion])
        edges = np.array([tup[1] for tup in expansion])
        return nodes, edges

    def _random_start(self):
        successors = None
        while not successors:
            next = str(self.g._i_to_node[np.random.randint(len(self.g.nodes))])
            successors = self.g.expand(next)
        return next
