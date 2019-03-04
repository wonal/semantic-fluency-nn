import nltk
from nltk.corpus import stopwords as nltk_stopwords

EXCEPTIONS = [
    "just",
    "most",
    "few",
    "against",
    "further",
]


class Corpus:
    def __init__(self, file_path="", exceptions=EXCEPTIONS):
        self.exceptions = exceptions
        self.lemmatizer = nltk.WordNetLemmatizer()
        self.tokenizer = nltk.RegexpTokenizer(r'[A-Za-z]+')

        self.raw_file_string = ""
        self.sentence_matrix = []
        self.vocab = set()
        self.filtered_vocab = set()
        self.stopwords = set()

        self._load_corpus(file_path)
        self._prepare_corpus(self.raw_file_string)
        self._find_unique_vocab(self.sentence_matrix)
        self._initialize_stopwords()
        self._create_filtered_vocab()

    def _load_corpus(self, file_path):
        with open(file_path, 'r', encoding="utf-8") as file:
            self.raw_file_string = file.read().replace('\n', ' ')

    def _prepare_corpus(self, corpus_string):
        sentences = nltk.sent_tokenize(corpus_string)
        self.sentence_matrix = [
            self.tokenizer.tokenize(sent) for sent in sentences
        ]
        self.sentence_matrix = [
            [self.lemmatizer.lemmatize(word.lower()) for word in sent]
            for sent in self.sentence_matrix
        ]

    def _find_unique_vocab(self, sentence_matrix):
        self.vocab = set()
        for sent in sentence_matrix:
            for wd in sent:
                self.vocab.add(wd)

    def _initialize_stopwords(self):
        self.stopwords = set(nltk_stopwords.words("english"))
        for wd in self.exceptions:
            self.stopwords.remove(wd)

    def _create_filtered_vocab(self):
        self.filtered_vocab = set(
            wd for wd in self.vocab if wd not in self.stopwords
        )
