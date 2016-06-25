# coding: utf-8

import numpy as np
from cv2.xfeatures2d import SIFT_create
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib

import os


class FeatureExtractor(object):
    def __init__(self, file=None):
        # initial cluster from file
        if os.path.isfile(file):
            self.cluster = joblib.load(file)
        else:
            self.cluster = None
        self.sift = SIFT_create()

    def is_initialized(self):
        if self.cluster:
            return True
        else:
            return False

    def train(self):
        if not self.is_initialized():
            self.retrain()

    def retrain(self, images, n_clusters=500, n_jobs=-1):
        descs = np.array([self.sift.detectAndCompute(img, None)[1] for img in images])
        # Sometimes descriptor is None, turn it into np.ndarray type.
        descs = [d if isinstance(d, np.ndarray) else
                np.array([]).reshape(0, 128).astype('float32') for d in descs]
        self.cluster = KMeans(n_clusters=n_clusters, n_jobs=n_jobs,
                              random_state=42).fit(np.vstack(descs))

    def save(self, path):
        joblib.dump(self.cluster, filename=path)
        print("Save to ", os.path.abspath(path), "Success!")

    def extract(self, image):
        # extract features from an image.
        assert self.cluster, "self.cluster should be initial!"
        n_clusters = self.cluster.n_clusters
        features = np.zeros(n_clusters)
        descriptors = self.sift.detectAndCompute(image, None)[1]
        if descriptors is None:
            return features
        y = self.cluster.predict(descriptors)
        features[list(set(y))] = 1
        return features


class Classifier(object):
    def __init__(self, file=None):
        # initial classifier from file
        if os.path.isfile(file):
            self.clf = joblib.load(file)
        else:
            self.clf = None

    def is_initialized(self):
        if self.clf:
            return True
        else:
            return False

    def train(self, X, y):
        # train classifier from given X, y.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                            random_state=42)
        params = {
            'C': [1e2, 3e2, 1e3, 3e3, 1e4],
            'gamma': [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
        }
        self.clf = GridSearchCV(SVC(), param_grid=params)
        self.clf.fit(X_train, y_train)
        print(self.clf.best_estimator_)
        y_pred = self.clf.predict(X_test)
        print(classification_report(y_test, y_pred))

    def save(self, path):
        joblib.dump(self.clf, filename=path)
        print("Save to ", os.path.abspath(path), "Success!")

    def classify(self, features):
        return self.clf.predict(features)


class Decision(object):
    pass
