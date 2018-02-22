import numpy as np
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import gensim
import re

"""
Module of tools to help with document classifications
"""


class LemmaTokenizer(object):
    def __init__(self):
         self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
         return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

def build_model(sentences, size):
    corpus_words = []
    for session in sentences:
        words = word_tokenize(session)
        session_words = []
        for word in words:
            regex_match = re.match(r"\b[a-zA-Z]+\b", word)
            if (regex_match is not None) & (len(word) > 1) & (word not in stopwords):
                session_words.append(wnl.lemmatize(word.lower()))
        corpus_words.append(session_words)
    model = gensim.models.Word2Vec(corpus_words, size=size, window=5, seed=42,
                                   min_count=2, workers=4)
    return model, corpus_words

def average_vectors(word_list, word2vec_dict, vec_length=5):
    avg_vector = np.zeros(vec_length)
    if len(word_list) == 0:
        return avg_vector
    for word in word_list:
        if word in word2vec_dict.keys():
            avg_vector = np.add(word2vec_dict[word], avg_vector)
    return np.divide(avg_vector, len(word_list))
