import numpy as np
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import gensim
import re
from sklearn.datasets import fetch_20newsgroups

"""
Module of tools to help with document classifications
"""


class LemmaTokenizer(object):
    def __init__(self):
         self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
         return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


def build_model(size, docs):
    """
    Builds word embeddings using word2vec and a custom corpus. Vectorizes
    words greater than one character, lemmatizes, and removes stopwords
    Args:
        size (int): The length of the word vectors
    Returns:
        model (gensim.model): The word2vec model
    """
    corpus_words = []
    for session in docs:
        words = word_tokenize(session)
        session_words = []
        for word in words:
            regex_match = re.match(r"\b[a-zA-Z]+\b", word)
            if (regex_match is not None) & (len(word) > 1) & (word not in stopwords):
                session_words.append(wnl.lemmatize(word.lower()))
        corpus_words.append(session_words)
    model = gensim.models.Word2Vec(corpus_words, size=size, window=5, seed=42,
                                min_count=2, workers=4)
    return model

def build_model_from_google():
    """
    Build word embeddings using the google news embeddings. Word vectors
    are of length 300
    Returns:
        model (gensim.model): The word2vec model
    """
    model = gensim.models.KeyedVectors.\
                                load_word2vec_format('../data/GoogleNews-vectors-negative300.bin.gz',
                                                        binary=True, limit=100000)
    return model

def average_docs(data, model, vec_length=5):
    """
    Numerically encode each document as the average of all its word vectors
    Args:
        model (gensim.model): word2vec model
    Returns:
        vectors (list): list of lists of average vectors
    """
    w2v = dict(zip(model.index2word, model.vectors))
    vectors = []
    for email in data:
        email_vector = average_vectors(email, w2v, vec_length)
        vectors.append(email_vector)
    return vectors

def average_vectors(word_list, word2vec_dict, vec_length=5):
    """
    Aggregate the text data by averagin the vectors
    Args:
        words_list (list): list of words to average
        word2vec_dict (dict): dictionary of word: vector key value pairs
        vec_length (int): length of the vector. Default value is 5
    Returns:
        (np array) Average of the word vectors
    """

    avg_vector = np.zeros(vec_length)
    if len(word_list) == 0:
        return avg_vector
    for word in word_list:
        if word in word2vec_dict.keys():
            avg_vector = np.add(word2vec_dict[word], avg_vector)
    return np.divide(avg_vector, len(word_list))

def get_newsgroups_data():
    """
    Retrieve twenty newsgroups data. Requires sklearn and a manual download
    of the data
    """
    categories = ['rec.motorcycles', 'rec.autos']
    newsgroups_train = fetch_20newsgroups(subset='train',
                                        categories=categories,
                                        remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test',
                                        remove=('headers', 'footers', 'quotes'),
                                        categories=categories)
    return newsgroups_train, newsgroups_test

def data_padder(self):
    raise NotImplementedError
