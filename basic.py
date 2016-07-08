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
from sklearn.decomposition import PCA


class FeatureExtractor(object):
    def __init__(self, method='sift', file=None):
        self.method = method
        self.red = None  # 该方法为降维的辅助函数
        if self.method == 'sift':
            self.train = self.sift_train
            self.extract = self.sift_extract
        elif self.method == 'lbp':
            self.train = self.lbp_train
            self.extract = self.lbp_extract
        if file is not None:
            self.load(file)

    def load(self, file):
        """从文件加载self.red"""
        self.red = joblib.load(file)

    def is_initialized(self):
        """判断self.red是否已被初始化"""
        if self.red:
            return True
        else:
            return False

    def sift_train(self, images, n_clusters=120, n_jobs=-1):
        """利用sift，训练特征提取器(训练KMeans)"""
        sift = SIFT_create()
        descs = np.array([sift.detectAndCompute(img, None)[1] for img in images])
        # Sometimes descriptor is None, turn it into np.ndarray type.
        descs = [d if isinstance(d, np.ndarray) else
                np.array([]).reshape(0, 128).astype('float32') for d in descs]
        # 训练好的聚类器放入self.red
        self.red = KMeans(n_clusters=n_clusters, n_jobs=n_jobs,
                          random_state=42).fit(np.vstack(descs))

    def sift_extract(self, image):
        """利用sift，提取特征"""
        assert self.red, "self.red should be initial!"
        n_clusters = self.red.n_clusters  # 聚类的数量
        features = np.zeros(n_clusters)   # 提取到的特征
        sift = SIFT_create()
        descriptors = sift.detectAndCompute(image, None)[1]
        if descriptors is None:  # 如果没有找到一个描述子，就返回全是0的数组
            return features
        y = self.red.predict(descriptors)  # 对描述子聚类
        features[list(set(y))] = 1  # 得到最终的特征
        return features

    def lbp_train(self, images, n_components=0.95):
        """利用lbp，训练特征提取器(训练PCA)"""
        X = np.array([]).reshape(0, 1536)
        for img in images:
            height, width = img.shape
            w = width // 2
            h = height // 3
            feature = np.array([])
            # 将图像分为6个区域，非别求这6个区域的lbp特征
            for i in range(2):
                for j in range(3):
                    cell = img[h * j:h * (j + 1), w * i:w * (i + 1)]
                    lbp = local_binary_pattern(cell, 8, 1)
                    histogram = np.zeros(256)
                    for pattern in lbp.ravel():  # 求得直方图
                        histogram[int(pattern)] += 1
                    histogram = (histogram - histogram.mean()) / histogram.std()  # 将直方图归一化
                    feature = np.hstack((feature, histogram))
            # 将6个区域的直方图拼接成一个直方图
            X = np.vstack((X, feature))

        # 因为得到的特征维度太高，因此利用PCA对其降维
        self.red = PCA(n_components=n_components)
        self.red.fit(X)

    def lbp_extract(self, image):
        # extract features from an image using lbp
        assert self.red, "self.red should be initial!"
        height, width = image.shape
        w = width // 2
        h = height // 3
        feature = np.array([])
        for i in range(2):
            for j in range(3):
                cell = image[h*j:h*(j+1), w*i:w*(i+1)]
                lbp = local_binary_pattern(cell, 8, 1)
                histogram = np.zeros(256)
                for pattern in lbp.ravel():
                    histogram[int(pattern)] += 1
                histogram = (histogram - histogram.mean()) / histogram.std()
                feature = np.hstack((feature, histogram))
        feature = self.red.transform(feature.reshape(1, -1))
        return feature.ravel()


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
        # train classifier from given X, y using SVM
        self.clf = SVC(C=C, gamma=gamma, random_state=42)
        self.clf.fit(X, y)

    def classify(self, features):
        return self.clf.predict(features)


class Decision(object):
    """利用滑动窗口会得到大量符合条件的窗口，因此要对它们进行归类，得到最终的区域"""
    def __init__(self, method='max'):
        if method == 'max':
            self.decide = self._max_cover
        elif method == 'min':
            self.decide = self._min_cover
        else:
            self.decide = self._average_cover

    def _max_cover(self, X, n):
        """最大覆盖"""
        cluster = KMeans(n, random_state=42, n_jobs=-1)
        X_new = np.vstack((X[:, 0] + X[:, 1], X[:, 2] + X[:, 3])).T
        y = cluster.fit_predict(X_new)
        result = []
        for i in range(n):
            y_start = X[y == i][:, 0].min()
            y_end = X[y == i][:, 1].max()
            x_start = X[y == i][:, 2].min()
            x_end = X[y == i][:, 3].max()
            result.append((y_start, y_end, x_start, x_end))
        return np.array(result).astype('int64')

    def _min_cover(self, X, n):
        """最小覆盖"""
        cluster = KMeans(n, random_state=42, n_jobs=-1)
        X_new = np.vstack((X[:, 0] + X[:, 1], X[:, 2] + X[:, 3])).T
        y = cluster.fit_predict(X_new)
        result = []
        for i in range(n):
            y_start = X[y == i][:, 0].max()
            y_end = X[y == i][:, 1].min()
            x_start = X[y == i][:, 2].max()
            x_end = X[y == i][:, 3].min()
            result.append((y_start, y_end, x_start, x_end))
        return np.array(result).astype('int64')

    def _average_cover(self, X, n):
        """平均覆盖"""
        cluster = KMeans(n, random_state=42, n_jobs=-1)
        X_new = np.vstack((X[:, 0] + X[:, 1], X[:, 2] + X[:, 3])).T
        y = cluster.fit_predict(X_new)
        result = []
        for i in range(n):
            y_start = X[y == i][:, 0].mean()
            y_end = X[y == i][:, 1].mean()
            x_start = X[y == i][:, 2].mean()
            x_end = X[y == i][:, 3].mean()
            result.append((y_start, y_end, x_start, x_end))
        return np.array(result).astype('int64')
