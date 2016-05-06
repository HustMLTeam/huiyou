# coding: utf-8
import numpy as np

from sklearn.cluster import KMeans


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

def min_cover(n):
    def dec(X):
        cluster = KMeans(n, random_state=42)
        y = cluster.fit_predict(X)
        result = np.array([]).reshape(0, 4)
        for i in range(n):
            y_start = X[y == i][:, 0].max()
            y_end = X[y == i][:, 1].min()
            x_start = X[y == i][:, 2].max()
            x_end = X[y == i][:, 3].min()
            result = np.vstack((result, (y_start, y_end, x_start, x_end)))
        return result
    return dec

def average_cover(n):
    def dec(X):
        cluster = KMeans(n, random_state=42)
        y = cluster.fit_predict(X)
        result = np.array([]).reshape(0, 4)
        for i in range(n):
            y_start = X[y == i][:, 0].sum() / X[y == i].shape(0)
            y_end = X[y == i][:, 1].sum() / X[y == i].shape(0)
            x_start = X[y == i][:, 2].sum() / X[y == i].shape(0)
            x_end = X[y == i][:, 3].sum() / X[y == i].shape(0)
            result = np.vstack((result, (y_start, y_end, x_start, x_end)))
        return result
    return dec
