import unittest
from src.graph.semantic_network import SemanticNetwork
import numpy as np


class SemanticGraphTests(unittest.TestCase):
    def setUp(self):
        np.random.seed(8)
        self.vectors = np.array([
            np.arange(0, 3),
            np.arange(3, 6),
            np.arange(6, 9),
            np.arange(9, 12),
            np.arange(12, 15),
            np.arange(15, 18),
            np.arange(21, 24),
            np.arange(24, 27),
            np.arange(27, 30),
        ])
        self.keys = ['key{:05d}'.format(i) for i in range(len(self.vectors))]
        self.network = SemanticNetwork(embeddings=self.vectors,
                                       aligned_keys=self.keys)

    def tearDown(self):
        self.vectors = None
        self.keys = None
        self.network = None

    @property
    def expected_embedding_matrix(self):
        expected = []
        for i in zip(self.keys, self.vectors):
            expected.append(i)
        return expected

    @property
    def actual_embedding_matrix(self):
        actual = []
        for i in range(len(self.network.aligned_keys)):
            actual.append(tuple((
                self.network.aligned_keys[i],
                self.network.embedding_matrix[i])))
        return actual

    def test_embedding_matrix_len(self):
        expected = len(self.expected_embedding_matrix)
        actual = len(self.actual_embedding_matrix)
        self.assertEqual(expected, actual)

    def test_stop_words_with_update(self):
        expected = self.network.graph.adjacency_matrix[:len(self.vectors),:len(self.vectors)].copy()
        self.network.update(em_proportion=1,
                            g_proportion=1,
                            stop_set=set(self.keys))
        actual = self.network.graph.adjacency_matrix[:len(self.vectors),:len(self.vectors)].copy()
        self.assertTrue((expected == actual).all())

