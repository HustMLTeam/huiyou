# coding: utf-8

import numpy as np
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.externals import joblib

import os

from load import load_tube, load_window
import feature_extractor


class Classifier(object):
    def __init__(self, feature='sift'):
        assert feature in ['sift', 'pca'], "feature should be 'sift' or 'pca'"
        if feature == 'sift':
            self.feature = feature_extractor.Sift()
        elif feature == 'pca':
            self.feature = feature_extractor.Pca()
        self.tube_file = None
        self.window_file = None

    def tube(self):
        if os.path.isfile(self.tube_file):
            clf = joblib.load(self.tube_file)
        else:
            X, y = load_tube()
            feature_extr = self.feature.tube()
            X_new = feature_extr(X)
            clf = self._train(X_new, y)
            joblib.dump(clf, self.tube_file)
        return clf.predict

    def window(self):
        if os.path.isfile(self.window_file):
            clf = joblib.load(self.window_file)
        else:
            X, y = load_window()
            feature_extr = self.feature.window()
            X_new = feature_extr(X)
            clf = self._train(X_new, y)
            joblib.dump(clf, self.window_file)
        return clf.predict

    def _train(self, X, y):
        pass


class Svm(Classifier):
    def __init__(self, feature='sift'):
        super(self.__class__, self).__init__(feature)
        self.tube_file = 'data/pkl/tube_svm_%s.pkl' % feature
        self.window_file = 'data/pkl/window_svm_%s.pkl' % feature

    def _train(self, X, y):
        params = {
            'C': [1e2, 3e2, 1e3, 3e3],
            'gamma': [1e-5, 3e-5, 1e-4, 3e-4]
        }
        clf = GridSearchCV(SVC(), param_grid=params)
        clf.fit(X, y)
        return clf
