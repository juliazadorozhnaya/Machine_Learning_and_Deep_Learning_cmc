import numpy as np
from collections import defaultdict


def kfold_split(num_objects, num_folds):
    res = []
    fold_size = num_objects // num_folds
    for i in range(num_folds - 1):
        train = np.array([j for j in range(i * fold_size)], dtype=int)
        res.append((np.concatenate((train, np.array([j for j in range((i + 1) * fold_size, num_objects)]))), np.array([j for j in range(i * fold_size, (i + 1) * fold_size)])))
    res.append((np.array([j for j in range((num_folds - 1) * fold_size)]), np.array([j for j in range((num_folds - 1) * fold_size, num_objects)])))
    return res



def knn_cv_score(X, y, parameters, score_function, folds, knn_class):
    answer = {}
    for normalizers in parameters['normalizers']:
        for N in parameters['n_neighbors']:
            for metrics in parameters['metrics']:
                for weights in parameters['weights']:

                    for f in folds:
                        if normalizers[0] is not None:
                            normalizers[0].fit(X[f[0]])
                            X_ = normalizers[0].transform(X)
                        else:
                            X_ = X
                    kNN = knn_class(n_neighbors=N, weights=weights, metric=metrics)
                    kNN.fit(X_[f[0]], y[f[0]])
                    tmp = 0
                    for i in range(len(folds)):
                        kNN.fit(X[folds[i][0]], y[folds[i][0]])
                        y_pred = kNN.predict(X[folds[i][1]])
                        s_tmp = score_function(y[folds[i][1]], y_pred)

                        tmp += s_tmp

                    tmp /= len(folds)
                    answer[(normalizers[1], N, metrics, weights)] = tmp

    return answer