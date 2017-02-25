# coding: utf-8
"""
trainer
=======
该模块为训练模块，利用已有的训练集训练出特征提取器以及分类器，并将结果保存在data/pkl文件夹下。

使用方法

  >>> t = Trainer()
  >>> t.train_window_extractor()
  Window extractor has been trained and saved to "data/pkl".
  >>> t.train_window_classifier()
  Window classifier has been trained and saved to "data/pkl".

"""
from basic import FeatureExtractor, Classifier

import os
import numpy as np
from scipy import misc
from sklearn.externals import joblib


class Trainer(object):
    """
    **训练器**

    利用basic模块里的类，对特征提取器以及分类器进行训练。

    """
    def __init__(self):
        self.window_x = None
        self.window_y = None
        self.window_extrct = FeatureExtractor()
        self.window_clf = Classifier()

    def train_window_extractor(self):
        """
        训练窗口特征提取器。
        """
        if self.window_x is None:
            self.load_window()  # 加载样本

        self.window_extrct.train(self.window_x) # 训练特征提取器
        if not os.path.exists('data/pkl'):
            os.makedirs('data/pkl')
        joblib.dump(self.window_extrct.red, 'data/pkl/window_lbp.pkl')  # 保存特征提取器
        print('Window extractor has been trained and saved to "data/pkl".')

    def train_window_classifier(self):
        """
        训练窗口分类器。
        """
        if self.window_x is None:
            self.load_window()  # 加载样本
        if not self.window_extrct.is_initialized():
            self.window_extrct.load('data/pkl/window_lbp.pkl')
        window_x = np.array([self.window_extrct.extract(img) for img in self.window_x]) # 从利用窗口特征提取器从样本中提取特征
        self.window_clf.train(window_x, self.window_y)  # 训练窗口分类器
        if not os.path.exists('data/pkl'):
            os.makedirs('data/pkl')
        joblib.dump(self.window_clf.clf, 'data/pkl/window_lbp_svm.pkl') # 保存窗口分类器
        print('Window classifier has been trained and saved to "data/pkl"')

    def load_window(self, posdir='data/window/pos', negdir='data/window/neg'):
        """
        加载窗口的正负样本。

        Parameters
        ----------
        posdir : str
            正样本所在的文件夹路径。

        negdir : str
            负样本所在的文件夹路径。
        """
        pos_files = [os.path.join(posdir, f) for f in os.listdir(posdir)]   # 所有正样本的完整路径名
        neg_files = [os.path.join(negdir, f) for f in os.listdir(negdir)]   # 所有负样本的完整路径名
        X_pos = np.array([misc.imread(f, mode='L') for f in pos_files])     # 读取正样本
        y_pos = np.ones(X_pos.shape[0])                                     # 对应的y置为1
        X_neg = np.array([misc.imread(f, mode='L') for f in neg_files])     # 读取负样本
        y_neg = np.zeros(X_neg.shape[0])                                    # 对应的y置为0
        self.window_x = np.concatenate((X_pos, X_neg))                      # 将正负样本合起来
        self.window_y = np.hstack((y_pos, y_neg))
