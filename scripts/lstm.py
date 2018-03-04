import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import imdb
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from tensorflow.contrib import learn
import classifier_helpers as ch

#fix the random seed for reproducibility
np.random.seed(7)


print("Loading data...")
train, test = ch.get_newsgroups_data()

# Build vocabulary
x_text = ch.lemmatize_strip(train.data)
max_document_length = max([len(x) for x in x_text])
x_text_flat = []
for sentence in x_text:
    sentence_as_list_item = ' '.join(sentence)
    x_text_flat.append(sentence_as_list_item)

vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text_flat)))
y = np.vstack(((train.target), (1-train.target))).T
X_train, X_test, y_train, y_test = train_test_split(x, y,
                                        test_size=0.2, random_state=42)


