import numpy as np
from sklearn.svm import SVC


def train_svm_and_predict(train_features, train_target, test_features):
    model = SVC(C =0.75, kernel = 'poly', degree =2)
    model.fit(train_features, train_target)
    return np.array(model.predict(test_features))