# coding: utf-8
import numpy as np

from sklearn.cluster import KMeans


#除了中心还有大小要考虑


def max_cover(n):
    def dec(X):
        cluster = KMeans(n, random_state=42)
        y = cluster.fit_predict(X)
        result = np.array([]).reshape(0, 4)
        for i in range(n):
            y_start = X[y == i][:, 0].min()
            y_end = X[y == i][:, 1].max()
            x_start = X[y == i][:, 2].min()
            x_end = X[y == i][:, 3].max()
            result = np.vstack((result, (y_start, y_end, x_start, x_end)))
        return result
    return dec

def Overlapping_cover(n):
    def dec(X):
        cluster = KMeans(n, random_state=42)
        y = cluster.fit_predict(X)
        result = np.array([]).reshape(0, 4)
        for i in range(n):
            y_start = X[y == i][:, 0].min()
            y_end = X[y == i][:, 1].max()
            x_start = X[y == i][:, 2].min()
            x_end = X[y == i][:, 3].max()
            result = np.vstack((result, (y_start, y_end, x_start, x_end)))
        return result
    return dec

