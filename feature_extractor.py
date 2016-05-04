# coding: utf-8

import numpy as np
from cv2.xfeatures2d import SIFT_create
from sklearn.cluster import KMeans
from sklearn.externals import joblib
import os

from load import load_tube, load_window


class Sift(object):
    def tube(self):
        file = 'data/pkl/tube_sift.pkl'
        n_clusters = 100
        if os.path.isfile(file):
            cluster = joblib.load(file)
        else:
            X, y = load_tube()
            cluster = self._train(X, n_clusters)
            joblib.dump(cluster, file)
        return lambda image: self.get_features(image, cluster)

    def window(self):
        file = 'data/pkl/window_sift.pkl'
        n_clusters = 100
        if os.path.isfile(file):
            cluster = joblib.load(file)
        else:
            X, y = load_window()
            cluster = self._train(X, n_clusters)
            joblib.dump(cluster, file)
        return lambda image: self.get_features(image, cluster)

    def _train(self, images, n_clusters):
        sift = SIFT_create()
        descs = np.array([sift.detectAndCompute(img, None)[1] for img in images])
        cluster = KMeans(n_clusters=n_clusters,
                         random_state=42).fit(np.vstack(descs))
        return cluster

    @staticmethod
    def get_features(image, cluster):
        sift = SIFT_create()
        descriptors = sift.detectAndCompute(image, None)[1]
        y = cluster.predict(descriptors)
        n_clusters = cluster.n_clusters
        features = np.zeros(n_clusters)
        features[list(set(y))] = 1
        return features


class Pca(object):
    pass
