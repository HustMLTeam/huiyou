# coding: utf-8
import numpy as np

from sklearn.cluster import KMeans


def max_cover(n):
    def dec(X):
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
        return np.array(result)
    return dec


def min_cover(n):
    def dec(X):
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
        return np.array(result)
    return dec


def average_cover(n):
    def dec(X):
        return X
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
        return np.array(result)
    return dec
