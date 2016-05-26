import codecs
from itertools import count
from keras.preprocessing import sequence
import sys
from nltk.tokenize.treebank import TreebankWordTokenizer
import numpy as np


def testset_read(fn, word_idx, maxlen):
    tokenizer = TreebankWordTokenizer()
    try:
        lines = codecs.open(fn, encoding='utf8').read().splitlines()
    except UnicodeDecodeError:
        lines = codecs.open(fn).read().splitlines()
    X = []
    for line in lines:
        s = []
        for token in tokenizer.tokenize(line):
            idx = word_idx.get(token, 1)  # 1 is UNKNOWN word id
            s.append(idx)
        X.append(s)

    X = sequence.pad_sequences(X, maxlen=maxlen)
    return X


def read(fn, test_percentage, maxlen, max_features, dataset_type):
    """
    :param fn: dataset filename.
    :param maxlen: maximum length for each sentence.
    :param max_features: max_features (e.g., unique words, vocabulary)
    :param padding: If true, padding will be made starting and ending of each sentence.
    :return:
    """

    tokenizer = TreebankWordTokenizer()
    c = count(2)
    word_idx = {}
    try:
        lines = codecs.open(fn, encoding='utf8').read().splitlines()
    except UnicodeDecodeError:
        lines = codecs.open(fn).read().splitlines()
    y = []
    X = []
    for line in lines:
        try:
            label, sentence = line.split('\t')
        except ValueError:
            continue
        y.append(label)
        s = []
        for token in tokenizer.tokenize(sentence):
            idx = word_idx.get(token, None)
            if idx is None:
                idx = c.next()
                if idx < max_features:
                    word_idx[token] = idx
                else:
                    idx = 1
            s.append(idx)
        X.append(s)

    X = sequence.pad_sequences(X, maxlen=maxlen)
    num_instance_for_train = int(len(X) * (1 - test_percentage))

    # convert labels into floats if the labels are real-valued.
    if dataset_type == 'regression':
        y = map(lambda e: float(e), y)
    else:
        set_y = set(y)
        print >> sys.stderr, set_y
        label1, label2 = set_y  # now supporting only binary classification.
        labels = {label1: 0, label2: 1}
        y = map(lambda e: labels[e], y)  # map labels 0/1.

    y = np.array(y)

    print "training set size {}, test set size {}".format(num_instance_for_train,
                                                          max(len(X) - num_instance_for_train, 0))

    return (X[:num_instance_for_train, :], y[:num_instance_for_train]), (X[num_instance_for_train:, :],
                                                                         y[num_instance_for_train:]), word_idx
