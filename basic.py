# coding: utf-8

import numpy as np
from skimage.feature import local_binary_pattern
from cv2.xfeatures2d import SIFT_create
from sklearn.cluster import KMeans
from sklearn.svm import SVC
# from sklearn.grid_search import GridSearchCV
# from sklearn.metrics import classification_report
# from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib


class FeatureExtractor(object):
    def __init__(self, method='sift', file=None):
        self.method = method
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
            pass

    def extract(self, image):
        if self.method == 'sift':
            return self.sift_extract(image)
        else:
            return self.lbp_extract(image)

    def sift_train(self, images, n_clusters=500, n_jobs=-1):
        sift = SIFT_create()
        descs = np.array([sift.detectAndCompute(img, None)[1] for img in images])
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
        sift = SIFT_create()
        descriptors = sift.detectAndCompute(image, None)[1]
        if descriptors is None:
            return features
        y = self.cluster.predict(descriptors)
        features[list(set(y))] = 1
        return features

    def lbp_extract(self, image):
        height, width = image.shape
        w = width // 2
        h = height // 3
        feature = np.array([])
        for i in range(2):
            for j in range(3):
                cell = image[h*j:h*(j+1), w*i:w*(i+1)]
                lbp = local_binary_pattern(cell, 20, 2, method='uniform')
                histogram = np.zeros(22)
                for pattern in lbp.ravel():
                    histogram[int(pattern)] += 1
                histogram = (histogram - histogram.mean()) / histogram.std()
                feature = np.hstack((feature, histogram))
        return feature


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

    def train(self, X, y, C=1000, gamma=0.003):
        # train classifier from given X, y.
        self.clf = SVC(C=C, gamma=gamma, random_state=42)
        self.clf.fit(X, y)

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
