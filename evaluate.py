""" Package contains performance evaluation methods.
"""
import sys
import data
from model import create_logistic_model, create_regression_model
from utils import cross_validate
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score


fn = sys.argv[1]
n_folds = int(sys.argv[2])
problem_type = sys.argv[3]

assert problem_type in ('regression', 'classification'), "Problem type should be either regression or classification"

maxlen = 20  # maximum length for each sentence.
max_features = 25000  # length of the vocabulary.
batch_size = 32
nb_epoch = 3


(X_train, y_train), (_, _) = data.read(fn, 0.0, maxlen, max_features, problem_type)
print >> sys.stderr, 'X_train shape:', X_train.shape

if problem_type == 'regression':
    model = create_regression_model(maxlen, max_features)
    cross_validate(model, X_train, y_train, n_folds, batch_size, nb_epoch, func_for_evaluation=pearsonr)
else:
    model = create_logistic_model(maxlen, max_features)
    cross_validate(model, X_train, y_train, n_folds, batch_size, nb_epoch, func_for_evaluation=accuracy_score)

