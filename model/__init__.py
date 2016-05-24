import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Lambda
from keras.layers import Embedding
from keras.layers import Convolution1D
from keras import backend as K
from representations import get_embedding_weights
import sys


# set parameters:
embedding_dims = 300
nb_filter = 250
filter_length = 3
hidden_dims = 250


def max_1d(X):
    return K.max(X, axis=1)


def __get_base_model(maxlen, max_features, word_idx, use_pretrained_embeddings=False):
    """
    :param maxlen: sentence size. Longer sentences will be truncated.
    :param max_features: vocab size.
    :param word_idx: {word1: index1, word2: index2}
    :return:
    """

    print >> sys.stderr, 'Build model...'
    model = Sequential()

    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions

    if use_pretrained_embeddings:
        print >> sys.stderr, 'Reading embeddings...'
        embedding_weights = get_embedding_weights(word_idx)
        model.add(Embedding(max_features,
                            embedding_dims,
                            input_length=maxlen,
                            dropout=0.2,
                            weights=[embedding_weights]))
    else:
        model.add(Embedding(max_features,
                            embedding_dims,
                            input_length=maxlen,
                            dropout=0.2))

    # we add a Convolution1D, which will learn nb_filter
    # word group filters of size filter_length:
    model.add(Convolution1D(nb_filter=nb_filter,
                            filter_length=filter_length,
                            border_mode='valid',
                            activation='relu',
                            subsample_length=1))

    # we use max over time pooling by defining a python function to use
    # in a Lambda layer

    model.add(Lambda(max_1d, output_shape=(nb_filter,)))

    # We add a vanilla hidden layer:
    model.add(Dense(hidden_dims))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))

    model.add(Dense(1))
    return model


def create_regression_model(maxlen, max_features, word_idx, use_pretrained_embeddings):
    model = __get_base_model(maxlen, max_features, word_idx, use_pretrained_embeddings)
    model.add(Activation('linear'))
    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def create_logistic_model(maxlen, max_features, word_idx, use_pretrained_embeddings):
    model = __get_base_model(maxlen, max_features, word_idx, use_pretrained_embeddings)
    model.add(Activation('softmax'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def train_model(model, X, y, batch_size, num_epoch):

    np.random.seed(42)
    idx = np.random.permutation(len(y))

    X = np.array(X)
    y = np.array(y)

    X = X[idx, :]
    y = y[idx]

    model.fit(X, y, batch_size=batch_size, nb_epoch=num_epoch)

