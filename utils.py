import codecs
from itertools import count
import numpy as np


def read_formality_dataset(max_features=5000):
    c = count(2)
    d = {}
    lines = codecs.open('formality.lahiri.dataset').read().splitlines()
    y = []
    X = []
    for line in lines:
        try:
            score, sentence = line.split('\t')
        except ValueError:
            continue
        y.append(float(score))
        s = []
        for token in sentence.split():
            idx = d.get(token, None)
            if idx is None:
                idx = c.next()
                if idx < max_features:
                    d[token] = idx
                else:
                    d[token] = 1
                    idx = 1
            s.append(idx)
        X.append(s)

    seed = 5
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    print len(X), len(y)

    return (X[:9000], y[:9000]), (X[9000:], y[9000:])
