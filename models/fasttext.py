'''Major portion of the code obtained from https://github.com/keras-team/keras/blob/master/examples/imdb_fasttext.py

FastText running over imdb dataset
Based on Joulin et al's paper:
Bags of Tricks for Efficient Text Classification
https://arxiv.org/abs/1607.01759
Results on IMDB datasets with uni and bi-gram embeddings:
    Uni-gram: 0.8813 test accuracy after 5 epochs. 8s/epoch on i7 cpu.
    Bi-gram : 0.9056 test accuracy after 5 epochs. 2s/epoch on GTx 980M gpu.
'''

import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D
from keras.datasets import imdb
from sklearn.metrics import precision_recall_fscore_support as score


def create_ngram_set(input_list, ngram_value = 2):
    """
    Extract a set of n-grams from a list of integers.
    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
    {(4, 9), (4, 1), (1, 4), (9, 4)}
    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
    [(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]
    """
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))


def add_ngram(sequences, token_indice, ngram_range = 2):
    """
    Augment the input list of list (sequences) by appending n-grams values.
    Example: adding bi-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
    >>> add_ngram(sequences, token_indice, ngram_range=2)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]
    Example: adding tri-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017, (7, 9, 2): 2018}
    >>> add_ngram(sequences, token_indice, ngram_range=3)
    [[1, 3, 4, 5, 1337], [1, 3, 7, 9, 2, 1337, 2018]]
    """
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for i in range(len(new_list) - ngram_range + 1):
            for ngram_value in range(2, ngram_range + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)

    return new_sequences

# Set parameters:
# ngram_range = 2 will add bi-grams features
ngram_range = 2
MAX_FEATURES = 20000
MAX_SEQUENCE_LENGTH = 400
batch_size = 32
embedding_dims = 50

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = MAX_FEATURES)

if ngram_range > 1:
    # Create set of unique n-gram from the training set.
    ngram_set = set()
    for input_list in x_train:
        for i in range(2, ngram_range + 1):
            set_of_ngram = create_ngram_set(input_list, ngram_value=i)
            ngram_set.update(set_of_ngram)

    # Dictionary mapping n-gram token to a unique integer.
    # Integer values are greater than MAX_FEATURES in order
    # to avoid collision with existing features.
    start_index = MAX_FEATURES + 1
    token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
    indice_token = {token_indice[k]: k for k in token_indice}

    # MAX_FEATURES is the highest integer that could be found in the dataset.
    MAX_FEATURES = np.max(list(indice_token.keys())) + 1

    # Augmenting x_train and x_test with n-grams features
    x_train = add_ngram(x_train, token_indice, ngram_range)
    x_test = add_ngram(x_test, token_indice, ngram_range)

x_train = sequence.pad_sequences(x_train, maxlen = MAX_SEQUENCE_LENGTH)
x_test = sequence.pad_sequences(x_test, maxlen = MAX_SEQUENCE_LENGTH)

# Building the network architecture
model = Sequential()
model.add(Embedding(MAX_FEATURES, embedding_dims, input_length = MAX_SEQUENCE_LENGTH))
model.add(GlobalAveragePooling1D())
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size = batch_size,
          epochs = 5,
          validation_data = (x_test, y_test))

# Evaluating results
predicted_result = model.predict_classes(x_test, batch_size = batch_size)

print("\n\n_________________________\nResult", y_test, '\n_________________________\n\n')
    
precision, recall, fscore, support = score(y_test, predicted_result)
count = 0
for i in range(len(y_test)):
    if(y_test[i] == predicted_result[i]):
        count+=1

print('accuracy: ', count/len(y_test))
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))

