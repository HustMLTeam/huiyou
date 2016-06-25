# coding: utf-8

import numpy as np
from cv2.xfeatures2d import SIFT_create
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib


class FeatureExtractor(object):
    def __init__(self, method='sift', file=None):
        self.method = method
        self.sift = SIFT_create()
        if file is None:
            self.cluster = None
        else:
            self.load_cluster(file)

    def load_cluster(self, file):
        self.cluster = joblib.load(file)

    def is_initialized(self):
        if self.cluster:
            return True
        else:
            return False

    def train(self, images, n_clusters=500):
        if self.method == 'sift':
            self.sift_train(images, n_clusters)
        else:
            self.lbp_train(images)

    def extract(self, image):
        if self.method == 'sift':
            return self.sift_extract(image)
        else:
            return self.lbp_extract(image)

    def sift_train(self, images, n_clusters=500, n_jobs=-1):
        descs = np.array([self.sift.detectAndCompute(img, None)[1] for img in images])
        # Sometimes descriptor is None, turn it into np.ndarray type.
        descs = [d if isinstance(d, np.ndarray) else
                np.array([]).reshape(0, 128).astype('float32') for d in descs]
        self.cluster = KMeans(n_clusters=n_clusters, n_jobs=n_jobs,
                              random_state=42).fit(np.vstack(descs))

    def sift_extract(self, image):
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

    def lbp_train(self, images):
        pass

    def lbp_extract(self, image):
        pass


class Classifier(object):
    def __init__(self, classify_method='svm', file=None):
        self.classify_method = classify_method
        # initial classifier from file
        if file:
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

    def classify(self, features):
        return self.clf.predict(features)


class Decision(object):
    def __init__(self, method='max'):
        if method == 'max':
            self.decide = self._max_cover
        elif method == 'min':
            self.decide = self._min_cover
        else:
            self.decide = self._average_cover

    def _max_cover(self, X, n):
        cluster = KMeans(n, random_state=42, n_jobs=-1)
        y = cluster.fit_predict(X)
        result = []
        for i in range(n):
            y_start = X[y == i][:, 0].min()
            y_end = X[y == i][:, 1].max()
            x_start = X[y == i][:, 2].min()
            x_end = X[y == i][:, 3].max()
            result.append((y_start, y_end, x_start, x_end))
        result.sort(key=lambda l: l[2])
        return np.array(result).astype('int64')

    def _min_cover(self, X, n):
        cluster = KMeans(n, random_state=42, n_jobs=-1)
        y = cluster.fit_predict(X)
        result = []
        for i in range(n):
            y_start = X[y == i][:, 0].max()
            y_end = X[y == i][:, 1].min()
            x_start = X[y == i][:, 2].max()
            x_end = X[y == i][:, 3].min()
            result.append((y_start, y_end, x_start, x_end))
        result.sort(key=lambda l: l[2])
        return np.array(result).astype('int64')

    def _average_cover(self, X, n):
        cluster = KMeans(n, random_state=42, n_jobs=-1)
        y = cluster.fit_predict(X)
        result = []
        for i in range(n):
            y_start = X[y == i][:, 0].mean()
            y_end = X[y == i][:, 1].mean()
            x_start = X[y == i][:, 2].mean()
            x_end = X[y == i][:, 3].mean()
            result.append((y_start, y_end, x_start, x_end))
        result.sort(key=lambda l: l[2])
        return np.array(result).astype('int64')
