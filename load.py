# coding: utf-8


import numpy as np
from scipy import misc

import os


def load_tube():
    posdir = 'data/tube/pos/'
    negdir = 'data/tube/neg/'
    return load_data(posdir, negdir)


def load_window():
    posdir = 'data/window/pos/'
    negdir = 'data/window/neg/'
    return load_data(posdir, negdir)


def load_data(posdir, negdir):
    pos_files = [os.path.join(posdir, f) for f in os.listdir(posdir)]
    neg_files = [os.path.join(negdir, f) for f in os.listdir(negdir)]
    X_pos = np.array([misc.imread(f, mode='L') for f in pos_files])
    y_pos = np.ones(X_pos.shape[0])
    X_neg = np.array([misc.imread(f, mode='L') for f in neg_files])
    y_neg = np.zeros(X_neg.shape[0])
    return np.concatenate((X_pos, X_neg)), np.hstack((y_pos, y_neg))
