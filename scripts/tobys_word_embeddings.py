from sklearn.datasets import fetch_20newsgroups

"""
To do:
    1. routine to find the maximum sentence length in a data set, then pad them
    2. 
"""

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

class 

def build_dataset(documents):
    words = []
    for document in documents:
        doc_words = document.split()
        words.append_doc(words)
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
