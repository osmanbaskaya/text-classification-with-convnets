
import numpy as np
from sklearn.cross_validation import KFold


def cross_validate(model, X, y, n_folds, batch_size, num_epoch, func_for_evaluation=None):

    # let's shuffle first.
    seed = 5
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    X = np.array(X)
    y = np.array(y)

    scores = np.zeros(n_folds)
    kf = KFold(len(y), n_folds=n_folds)
    for i, (train_index, test_index) in enumerate(kf):
        X_train, y_train = X[train_index, :], y[train_index]
        X_test, y_test = X[test_index, :], y[test_index]
        model.fit(X_train, y_train,
                  batch_size=batch_size,
                  nb_epoch=num_epoch)

        predictions = model.predict(X_test)
        score = func_for_evaluation(predictions[:, 0].tolist(), y_test)
        try:
            scores[i] = score[0]
        except IndexError:
            scores[i] = score


    print "{}-Fold cross validation score: {}".format(n_folds, scores.mean())


