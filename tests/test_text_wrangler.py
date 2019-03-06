import unittest
from src.text.text_wrangler import Corpus
import os
shakespeare_dir = os.path.dirname(os.path.dirname(__file__))
shakespeare_path = os.path.join(shakespeare_dir, 'docs/shakespeare.txt')
corpus_sentences = 99624
vocab_size = 22444
filtered_vocab_size = 22316


class TestTextWrangler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shakespeare = Corpus(shakespeare_path)

    def test_sizes_are_correct(self):
        self.assertTrue(len(self.shakespeare.sentence_matrix), corpus_sentences)
        self.assertTrue(len(self.shakespeare.vocab), vocab_size)
        self.assertTrue(len(self.shakespeare.filtered_vocab), filtered_vocab_size)

    def test_exceptions_not_in_stopwords(self):
        for word in self.shakespeare.exceptions:
            self.assertTrue(word not in self.shakespeare.stopwords)

    def test_stopwords_not_in_filtered_vocabulary(self):
        for word in self.shakespeare.stopwords:
            self.assertTrue(word not in self.shakespeare.filtered_vocab)
