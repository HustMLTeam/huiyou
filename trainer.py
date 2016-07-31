# coding: utf-8
"""
trainer
=======
该模块为训练模块，利用已有的训练集训练出特征提取器以及分类器，并将结果保存在data/pkl文件夹下。

要训练寻找液位仪的位置

  >>> t = Trainer(extract_method='sift')
  >>> t.train_tube_extractor()
  Tube extractor has been trained and saved to "data/pkl".
  >>> t.train_tube_classifier()
  Tube classifier has been trained and saved to "data/pkl".

要训练寻找液位仪小窗口的位置

  >>> t = Trainer(extract_method='lbp')
  >>> t.train_window_extractor()
  Window extractor has been trained and saved to "data/pkl".
  >>> t.train_window_classifier()
  Window classifier has been trained and saved to "data/pkl".

"""
from basic import FeatureExtractor, Classifier

import os
import numpy as np
from scipy import misc
import cv2
from sklearn.externals import joblib


class Trainer(object):
    """
    **训练器**

    利用basic模块里的类，对特征提取器以及分类器进行训练。

    Parameters
    ----------
    extract_method : str
        特征提取的方法，可以为'sift'或'lbp'。

    classify_method : str
        分类的方法，目前只能选'svm'。

    """
    def __init__(self, extract_method='sift', classify_method='svm'):
        self.extract_method = extract_method
        self.classify_method = classify_method
        self.tube_x = None
        self.tube_y = None
        self.window_x = None
        self.window_y = None
        self.tube_extrct = FeatureExtractor(extract_method)
        self.window_extrct = FeatureExtractor(extract_method)
        self.tube_clf = Classifier(classify_method)
        self.window_clf = Classifier(classify_method)

    def train_tube_extractor(self):
        """
        训练液位仪特征提取器。
        """
        if self.tube_x is None:
            self.load_tube()
        self.tube_extrct.train(self.tube_x, 500)
        joblib.dump(self.tube_extrct.red, 'data/pkl/tube_%s.pkl' % self.extract_method)
        print('Tube extractor has been trained and saved to "data/pkl".')

    def train_window_extractor(self):
        """
        训练窗口特征提取器。
        """
        if self.window_x is None:
            self.load_window()

        self.window_extrct.train(self.window_x)
        joblib.dump(self.window_extrct.red, 'data/pkl/window_%s.pkl' % self.extract_method)
        print('Window extractor has been trained and saved to "data/pkl".')

    def train_tube_classifier(self):
        """
        训练液位仪分类器。
        """
        if self.tube_x is None:
            self.load_tube()
        if not self.tube_extrct.is_initialized():
            self.tube_extrct.load('data/pkl/tube_%s.pkl' % self.extract_method)
        tube_X = np.array([self.tube_extrct.extract(img) for img in self.tube_x])
        self.tube_clf.train(tube_X, self.tube_y)
        joblib.dump(self.tube_clf.clf, 'data/pkl/tube_%s_%s.pkl' %
                    (self.extract_method, self.classify_method))
        print('Tube classifier has been trained and saved to "data/pkl".')

    def train_window_classifier(self):
        """
        训练窗口分类器。
        """
        if self.window_x is None:
            self.load_window()
        if self.window_extrct.is_initialized():
            self.window_extrct.load('data/pkl/window_%s.pkl' % self.extract_method)
        window_x = np.array([self.window_extrct.extract(img) for img in self.window_x])
        self.window_clf.train(window_x, self.window_y)
        joblib.dump(self.window_clf.clf, 'data/pkl/window_%s_%s.pkl' %
                    (self.extract_method, self.classify_method))
        print('Window classifier has been trained and saved to "data/pkl"')

    def load_tube(self):
        """
        加载液位仪的正负样本。
        """
        self.tube_x, self.tube_y = self.load_data('data/tube/pos', 'data/tube/neg')

    def load_window(self):
        """
        加载窗口的正负样本。
        """
        self.window_x, self.window_y = self.load_data('data/window/pos', 'data/window/neg')
        # 直方图均衡化
        if self.extract_method == 'sift':
            self.window_x = np.array([cv2.equalizeHist(img) for img in self.window_x])

    def load_data(self, posdir, negdir):
        """
        加载样本。

        Parameters
        ----------
        posdir : str
            正样本所在的文件夹路径。

        negdir : str
            负样本所在的文件夹路径。
        """
        pos_files = [os.path.join(posdir, f) for f in os.listdir(posdir)]
        neg_files = [os.path.join(negdir, f) for f in os.listdir(negdir)]
        X_pos = np.array([misc.imread(f, mode='L') for f in pos_files])
        y_pos = np.ones(X_pos.shape[0])
        X_neg = np.array([misc.imread(f, mode='L') for f in neg_files])
        y_neg = np.zeros(X_neg.shape[0])
        return np.concatenate((X_pos, X_neg)), np.hstack((y_pos, y_neg))
