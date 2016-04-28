
# coding: utf-8


import numpy as np
from scipy import misc

from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.grid_search import GridSearchCV

from sklearn.externals import joblib

import os


tube_size = (200, 30)
window_size = (25, 10)


def load_tube(posdir='../data/tube/pos/', negdir='../data/tube/neg/'):
    return load_data(posdir, negdir, tube_size)


def load_window(posdir='../data/window/pos/', negdir='../data/window/neg/'):
    return load_data(posdir, negdir, window_size)


def load_data(posdir, negdir, size):
    pos_files = [os.path.join(posdir, f) for f in os.listdir(posdir)]
    neg_files = [os.path.join(negdir, f) for f in os.listdir(negdir)]
    X_pos = np.array([misc.imresize(misc.imread(f), size).ravel() for f in pos_files])
    y_pos = np.ones(X_pos.shape[0])
    X_neg = np.array([misc.imresize(misc.imread(f), size).ravel() for f in neg_files])
    y_neg = np.zeros(X_neg.shape[0])
    return np.vstack((X_pos, X_neg)), np.hstack((y_pos, y_neg))


def pca_svm(X, y, n_components=0.97):
    pca = PCA(n_components=n_components)
    X_new = pca.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.3, random_state=42)
    
    params = {'C': [1e3, 3e3],
          'gamma': [1e-5, 3e-5, 1e-4]
    }
    clf = GridSearchCV(SVC(class_weight='balanced'), params)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print(classification_report(y_test, pred))
    return pca, clf


if __name__ == '__main__':
    X, y = load_tube()
    pca, clf = pca_svm(X, y)
    X, y = load_window()
    pca, clf = pca_svm(X, y)