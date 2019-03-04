import unittest
from src.text.text_wrangler import Corpus
shakespeare_path = "../docs/shakespeare.txt"
corpus_sentences = 99624
vocab_size = 22444
filtered_vocab_size = 22316


class TestTextWrangler(unittest.TestCase):
    def test_sizes_are_correct(self):
        shakespeare = Corpus(shakespeare_path)
        self.assertTrue(len(shakespeare.sentence_matrix), corpus_sentences)
        self.assertTrue(len(shakespeare.vocab), vocab_size)
        self.assertTrue(len(shakespeare.filtered_vocab), filtered_vocab_size)

    def test_exceptions_not_in_stopwords(self):
        shakespeare = Corpus(shakespeare_path)
        for word in shakespeare.exceptions:
            self.assertTrue(word not in shakespeare.stopwords)

    def test_stopwords_not_in_filtered_vocabulary(self):
        shakespeare = Corpus(shakespeare_path)
        for word in shakespeare.stopwords:
            self.assertTrue(word not in shakespeare.filtered_vocab)
