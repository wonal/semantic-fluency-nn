import nltk


def load_corpus(file_path):
    file_str = ""
    with open(file_path, 'r', encoding="utf-8") as file:
        file_str = file.read().replace('\n', ' ')
    return file_str


def prepare_corpus(corpus_string):
    sentences = nltk.sent_tokenize(corpus_string)
    tokenizer = nltk.RegexpTokenizer(r'[A-Za-z]+')
    lemmatizer = nltk.WordNetLemmatizer()
    sentences_2d = [tokenizer.tokenize(sent) for sent in sentences]
    sentences_2d = [[lemmatizer.lemmatize(word.lower()) for word in sent]
                    for sent in sentences_2d]
    return sentences_2d


def unique_vocabulary(sentence_matrix):
    pass