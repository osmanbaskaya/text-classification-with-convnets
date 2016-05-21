""" Package contains performance evaluation methods.
"""
import sys
import data
from model import create_logistic_model, create_regression_model
from utils import cross_validate
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score


fn = sys.argv[1]
n_folds = int(sys.argv[2])
problem_type = sys.argv[3]
use_pretrained_embeddings = True if sys.argv[4].lower() == 'true' else False


print >> sys.stderr, fn, n_folds, problem_type, use_pretrained_embeddings


assert problem_type in ('regression', 'classification'), "Problem type should be either regression or classification"

maxlen = 20  # maximum length for each sentence.
max_features = 25000  # length of the vocabulary.
batch_size = 32
nb_epoch = 3
additional_num_words = 2  # "UNK" and "PADDING"


(X_train, y_train), (_, _), word_idx = data.read(fn, 0.0, maxlen, max_features, problem_type)
print >> sys.stderr, 'X_train shape:', X_train.shape

max_features = min(max_features, len(word_idx) + additional_num_words)

if problem_type == 'regression':
    model = create_regression_model(maxlen, max_features, word_idx, use_pretrained_embeddings)
    cross_validate(model, X_train, y_train, n_folds, batch_size, nb_epoch, func_for_evaluation=spearmanr)
else:
    model = create_logistic_model(maxlen, max_features, word_idx, use_pretrained_embeddings)
    cross_validate(model, X_train, y_train, n_folds, batch_size, nb_epoch, func_for_evaluation=accuracy_score)

