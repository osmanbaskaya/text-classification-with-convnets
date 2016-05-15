from gensim.models import Word2Vec
import numpy as np


WORD2VEC_PATH = 'GoogleNews-vectors-negative300.bin.gz'
WORD2VEC_EMB_DIMENSION = 300


def get_word2vec_model(path=WORD2VEC_PATH):
    return Word2Vec.load_word2vec_format(path)


def get_embedding_weights(vocab, path=WORD2VEC_PATH):
    model = get_word2vec_model(path)
    embedding_weights = np.random.rand(len(vocab) + 1, WORD2VEC_EMB_DIMENSION)
    embedding_weights[0, :] = np.zeros(WORD2VEC_EMB_DIMENSION)  # zeros for unk token.
    for word, idx in vocab.iteritems():
        if word in model:
            embedding_weights[idx, :] = model[word]
        else:
            print u"{} not found.".format(word)

    return embedding_weights
