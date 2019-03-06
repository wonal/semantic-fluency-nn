import unittest
from src.graph.semantic_network import SemanticNetwork
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SemanticGraphTests(unittest.TestCase):
    def setUp(self):
        np.random.seed(8)
        self.vectors = np.array([[np.random.randint(1000) for i in range(100)] for j in range(100)])
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

    @property
    def curr_adj_mat_copy(self):
        return self.network.graph.adjacency_matrix[:len(self.vectors),:len(self.vectors)].copy()

    def test_embedding_matrix_len(self):
        expected = len(self.expected_embedding_matrix)
        actual = len(self.actual_embedding_matrix)
        self.assertEqual(expected, actual)

    def test_update_with_all_vocab_as_stop_words(self):
        expected = self.curr_adj_mat_copy
        self.network.update(em_proportion=1,
                            g_proportion=1,
                            stop_set=set(self.keys))
        actual = self.curr_adj_mat_copy
        self.assertTrue((expected == actual).all())

    def test_update_with_include_set_is_stop_set(self):
        expected = self.curr_adj_mat_copy
        self.network.update(em_proportion=1,
                            g_proportion=1,
                            stop_set=set(self.keys))
        actual = self.curr_adj_mat_copy
        self.assertTrue((expected == actual).all())

    def test_update_with_empty_include_set(self):
        expected = self.curr_adj_mat_copy
        self.network.update(em_proportion=1,
                            g_proportion=1,
                            include_set=set())
        actual = self.curr_adj_mat_copy
        self.assertTrue((expected == actual).all())

    def test_update_cos_sim_correct(self):
        expected = self.curr_adj_mat_copy
        indices_to_change = range(len(self.vectors))
        thresh = 0.8
        cos_sim = [x[0][0] for x in [cosine_similarity([self.vectors[i]], [self.vectors[j]]) for i in indices_to_change for j in indices_to_change if i != j] if x[0][0] >= thresh]
        self.network.update(em_proportion=1,
                            g_proportion=1,
                            thresh=thresh,
                            include_set=set([self.keys[i] for i in indices_to_change]))
        actual = self.curr_adj_mat_copy
        changed_indx = np.where(expected != actual)
        self.assertTrue(np.allclose(cos_sim, self.network.graph.adjacency_matrix[changed_indx]))

