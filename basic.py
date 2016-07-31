# coding: utf-8
"""
basic
=====

该模块提供基本的类与函数，供detect以及trainer模块调用。这些类与函数主要是算法方
面的。
"""

import numpy as np
from scipy.signal import convolve2d
from skimage.feature import local_binary_pattern
from cv2.xfeatures2d import SIFT_create
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.decomposition import PCA


class FeatureExtractor(object):
    """
    **特征提取器**

    在从图片中寻找合适的液位仪以及窗口时，应先用滑动窗口得到一个个小图片，然后
    对这些图片提取特征才能进行分类。

    特征提取的方法为SIFT（尺度不变特征变换）或LBP（局部二值模式）。经过实验，如
    果是寻找大的液位仪，那么应当使用SIFT；如果是寻找液位仪上的小窗口，那么应当
    使用LBP。

    在实际应用之前要先对特征提取器利用train方法进行训练，用sift提取特征要利用
    kmeans将所得到的描述子聚类得到直方图，用lbp的话要利用pca对得到的直方图进行
    降维。训练好后要把结果保存在./data文件夹。（训练只用调用trainer里的相关方法
    就行了）

    如果已经训练好了并保存在了文件中，初始化时应指定相关文件。提取特征使用
    extract方法。

    Parameters
    ----------
    method : str
        特征提取的方法，可以为'sift'或'lbp'。如果为'sift'将提取sift描述子作为特
        征向量；如果为'lbp'将提取lbp特征作为特征向量。

    file : str，可选
        指定特定的文件从而加载特征提取器。如果是要训练特征提取器，可以省略这一
        参数。

    Warning
    -------
    不要调用sift_train与lbp_train，用train代替；也不要调用sift_extract与
    lbp_extract，用extract代替。

    Methods
    -------
    train(...)
        训练特征提取器，根据method的不同，该方法是sift_train或lbp_train的别名。
        在使用时调用该方法而不要调用sift_train或lbp_train。
    
    extract(...)
        提取特征，根据method的不同，该方法是sift_extract或lbp_extract的别名。
        在使用时调用该方法而不要调用sift_extract或lbp_extract。
    
    """
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
        """
        从文件加载特征提取器。

        Parameters
        ----------
        file : str
            指定特征提取器所在的文件。
        """
        self.red = joblib.load(file)

    def is_initialized(self):
        """
        判断特征提取器是否已被初始化。

        Returns
        -------
        bool
            如果已被初始化，返回True，否则返回False。
        """
        if self.red:
            return True
        else:
            return False

    def sift_train(self, images, n_clusters=120, n_jobs=-1):
        """
        利用SIFT，训练特征提取器(训练KMeans)

        只利用SIFT得到描述子是不够的，因为每幅图片的描述子维度太高而且数量不一
        致，所以利用词袋模型将这些描述子聚类得到直方图。该方法就是训练聚类器，
        以便能从描述子得到最终的直方图。    

        Parameters
        ----------
        images : 列表
            要用来训练的图片的集合。列表中的每个图片都是二维的numpy数组（灰度
            图）。

        c_clusters : int
            描述子聚类的类数，即特征向量的维度。

        n_jobs : int
            训练时用到的CPU核心数，如果是-1则使用全部核心。
        """
        sift = SIFT_create()
        descs = np.array([sift.detectAndCompute(img, None)[1] for img in images])
        # Sometimes descriptor is None, turn it into np.ndarray type.
        descs = [d if isinstance(d, np.ndarray) else
                np.array([]).reshape(0, 128).astype('float32') for d in descs]
        # 训练好的聚类器放入self.red
        self.red = KMeans(n_clusters=n_clusters, n_jobs=n_jobs,
                          random_state=42).fit(np.vstack(descs))

    def sift_extract(self, image):
        """
        利用SIFT，对给定的图片提取特征向量。使用前必须先初始化特征提取器。

        Parameters
        ----
        image : 二维numpy数组
            灰度图。

        Returns
        -------
        一维numpy数组
            图片的特征向量。
        """
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
        """
        利用LBP，训练特征提取器(训练PCA)

        将图像均分为6个区域（左上、右上、左中、右中、左下、右下），对这6个区域
        分别求LBP特征，然后根据这些特征出现的频率得到256维的直方图，把这6个直方
        图合并起来将得到1536维的特征向量。因为维数太高，在不损失准确度的前提下
        有必要利用PCA对其降维以提高运算速度。该方法就是训练PCA以便从1536维的特
        征向量得到较低维的特征向量。

        Parameters
        ----------
        images : 列表
            要用来训练的图片的集合。列表中的每个图片都是二维的numpy数组（灰度图）。

        n_components : int或float
            要保留的特征数。
            如果是大于1的整数，那么就保留n_components个特征；如果是小于1的浮点
            数，那么就保留n_components的方差。
        """
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
        """
        利用LBP，对给定的图片提取特征向量。使用前必须先初始化特征提取器。

        Parameters
        ----------
        image : 二维numpy数组
            灰度图的二维numpy数组。

        Returns
        -------
        一维numpy数组
            图片的特征向量。
        """
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
    """
    **分类器**

    利用特征提取器提取到特征向量后，再利用分类器判断是否为想要的图片。

    分类的方法目前只能使用SVM。

    在实际应用之前要先对特征提取器利用train方法进行训练，训练好后要把结果保存在
    ./data文件夹。（训练只用调用trainer里的相关方法就行了）

    如果已经训练好了并保存在了文件中，初始化时应指定相关文件。分类使用classify
    方法。

    Parameters
    ----------
    classify_method : str
        特征提取的方法，如果为'svm'将利用SVM作为分类的方法。目前只能使用SVM作为
        分类方法。

    file : str
        指定特定的文件从而加载分类器。如果是要训练特征提取器，可以省略这一参数。
    """
    def __init__(self, classify_method='svm', file=None):
        self.classify_method = classify_method
        # initial classifier from file
        if file:
            self.clf = joblib.load(file)
        else:
            self.clf = None

    def is_initialized(self):
        """
        判断分类器是否已被初始化。如果已被初始化，返回True，否则返回False。

        Returns
        -------
        bool
            如果已被初始化，返回True，否则返回False。
        """
        if self.clf:
            return True
        else:
            return False

    def train(self, X, y):
        """
        训练分类器

        利用给定的训练集训练分类器，使用SVM算法。

        Parameters
        ----------
        X : 二维numpy数组
            训练集的特征，为m行n列的矩阵，每一行为n维的特征向量，共有m组训练数据。

        y : 一维numpy数组
            训练集的目标，是长度为m的的数组，每个元素为0或1.
        """
        # train classifier from given X, y using SVM
        self.clf = SVC(C=1000, gamma=0.003, random_state=42)
        self.clf.fit(X, y)

    def classify(self, features):
        """
        分类

        对给定的特征向量分类。

        Parameters
        ----------
        features : 二维numpy数组
            每行为待分类的特征向量。

        Returns
        -------
        一维numpy数组
            每个值对应相应的特征向量，值为0或1。
        """
        return self.clf.predict(features)


class Decision(object):
    """
    **决策器**

    利用滑动窗口会得到大量符合条件的窗口，因此要对它们归类后进行相应的决策，才
    能得到最终的区域。
    
    先将所给的位置进行聚类，之后利用所选的决策方法得到最终的区域。
    有三种决策方法可选：max（最大覆盖），min（最小覆盖）和average（平均覆盖）。
    
    Parameters
    ----------
    method : str
        决策的方法。可以为'max'或'min'或'average'。

    Methods
    -------
    decide(X, n)
        从大量的区域决策得到最终的区域。

        Parameters
        ----------
        X : 二维numpy数组
            所要决策的位置的集合，每一行为一个区域，格式为(y_start, y_end, 
            x_start, x_end)

        n : int
            最终得到的位置的个数。
    """
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
            result.append((int(y_start), int(y_end), int(x_start), int(x_end)))
        return result

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
            result.append((int(y_start), int(y_end), int(x_start), int(x_end)))
        return result

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
            result.append((int(y_start), int(y_end), int(x_start), int(x_end)))
        return result


def horizontal_filter(src):
    """
    横向滤波。对二值化（0,255）的灰度图进行处理，得到新的二值化的灰度图。
    如果某像素点横向连续有4个都是255，那么该像素点对应的点的值为255，否则为0。

    Parameters
    ----------
    src : 二维numpy数组
        二值化的灰度图。

    Returns
    -------
    二维numpy数组
        经过处理后的图像。
    """
    result = convolve2d(src, np.ones((1, 4)) / 4, mode='same')
    result = np.where(result > 250, 255, 0).astype('uint8')
    return result
