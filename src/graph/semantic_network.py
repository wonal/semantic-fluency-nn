import numpy as np

from src.graph.Graph import UndirectedGraph


class SemanticNetwork:
    def __init__(self, model):
        self.model = model
        self.embedding_matrix = self.model.wv.vectors
        self.word_to_index = {}
        self.index_to_word = {}

    def _init_maps(self):
        for k, v in self.model.wv.vocab.items():
            self.word_to_index[k] = v.index
            self.index_to_word[v.index] = k


