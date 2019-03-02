import unittest
from graph.Graph import UndirectedGraph


class TestGraphMethods(unittest.TestCase):

    def setUp(self):
        self.g = UndirectedGraph()

    def tearDown(self):
        self.g = None

    def test_edge_none(self):
        node_1 = 1
        node_2 = 2

        expected = None
        actual = self.g.edge(node_1, node_2)

        self.assertEqual(expected, actual)

    def test_edge(self):
        node_1 = 1
        node_2 = 2
        edge = 3
        self.g.add_edge(node_1, node_2, edge)

        expected = edge
        actual = self.g.edge(node_1, node_2)

        self.assertEqual(expected, actual)

    def test_expansion_none(self):
        node = 1
        self.g.add_node(node)

        expected = []
        actual = self.g.expand(node)

        self.assertEqual(expected, actual)

    def test_expansion(self):
        node_1 = 1
        node_2 = 2
        node_3 = 3
        edge_1_2 = 4
        edge_1_3 = 5
        self.g.add_edge(node_1, node_2, edge_1_2)
        self.g.add_edge(node_1, node_3, edge_1_3)

        expected = [(node_2, edge_1_2), (node_3, edge_1_3)]
        actual = self.g.expand(1)

        self.assertEqual(expected, actual)

    def test_add_edge(self):
        node_1 = 1
        node_2 = 2
        edge = 3
        self.g.add_edge(node_1, node_2, edge)

        expected = 3
        actual = self.g.edge(1, 2)

        self.assertEqual(expected, actual)
