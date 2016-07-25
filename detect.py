# coding: utf-8

from basic import FeatureExtractor, Classifier, Decision, horizontal_filter
from slide_window import slide_window

import numpy as np
from scipy.signal import convolve2d
import cv2


class Locator(object):
    def __init__(self, image, tube_extract='sift', tube_classify='svm',
                 window_extract='lbp', window_classify='svm'):
        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        # 确保参数正确，参数为特征提取和分类的方法
        assert tube_extract in ['sift', 'lbp'], 'Tube extract method should be "sift" or "lbp"'
        assert window_extract in ['sift', 'lbp'], 'Window extract method should be "sift" or "lbp"'
        assert tube_classify in ['svm'], 'Tube classify method should be "svm"'
        assert window_classify in ['svm'], 'Window classify method should be "svm"'

        self.tube_extract_method = tube_extract
        self.tube_classify_method = tube_classify
        self.window_extract_method = window_extract
        self.window_classify_method = window_classify

        # 加载相应的特征提取器及分类器
        self.tube_extractor = FeatureExtractor(tube_extract, file='data/pkl/tube_%s.pkl' % tube_extract)
        self.tube_classifier = Classifier(tube_classify, file='data/pkl/tube_%s_%s.pkl' % (tube_extract, tube_classify))
        self.window_extractor = FeatureExtractor(window_extract, file='data/pkl/window_%s.pkl' % window_extract)
        self.window_classifier = Classifier(window_classify, file='data/pkl/window_%s_%s.pkl' % (window_extract, window_classify))

        self.tube = []
        self.window = []
        self.scale = []

    # 在图片上定位液位仪
    def locate_tube(self):
        positions = [] # 保存液位仪的位置，格式为(y_start, y_end, x_start, x_end)
        height, width = self.image.shape
        for y_start, y_end, x_start, x_end in slide_window(140, height,
                                width_min=22, width_max=32, width_inc=3,
                                height_min=200, height_max=240, height_inc=5,
                                ratio_min=7, ratio_max=9):
            feature = self.tube_extractor.extract(self.image[y_start:y_end, x_start:x_end])  # 提取特征
            if self.tube_classifier.classify(feature.reshape(1, -1)):  # 进行分类
                positions.append((y_start, y_end, x_start, x_end))
        # 在大量找到的区域中选出合适的结果
        dec = Decision('max')
        result = dec.decide(np.array(positions), 2)
        self.tube = sorted(result, key=lambda t: t[2])

    # 在液位仪上定位窗口
    def locate_window(self):
        for i, (y1, y2, x1, x2) in enumerate(self.tube):
            positions = []
            height = y2 - y1
            width = x2 - x1
            for y_start, y_end, x_start, x_end in slide_window(width, height,
                                    width_min=13, width_max=25, width_inc=3,
                                    height_min=26, height_max=39, height_inc=3,
                                    x_step=2, y_step=2):
                y_start += y1
                y_end += y1
                x_start += x1
                x_end += x1
                img = self.image[y_start:y_end, x_start:x_end]
                feature = self.window_extractor.extract(img)
                if self.window_classifier.classify(feature.reshape(1, -1)):
                    positions.append((y_start, y_end, x_start, x_end))
            dec = Decision('average')
            n_clusters = 5 if i == 0 else 7
            result = dec.decide(np.array(positions), n_clusters)
            self.window.append(sorted(result, key=lambda t: t[0]))

    def locate_scale(self):
        for y1, y2, x1, x2 in self.tube:
            ruler = self.image[y1:y2, x1+15:x2+15]

            conv = convolve2d(ruler, np.array([
                        [1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [-4, -4, -4, -4, -4, -4, -4, -4],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1]]) / 64, mode='same') - \
                   np.abs(convolve2d(ruler, np.array([
                       [1, 1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1, 1],
                       [0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0],
                       [-1, -1, -1, -1, -1, -1, -1, -1],
                       [-1, -1, -1, -1, -1, -1, -1, -1]]) / 32, mode='same'))

            filtered = horizontal_filter(np.where(conv > 15, 255, 0))
            upper_scale = np.median(np.where(filtered[75:125] == 255)[0]) + 75
            lower_scale = np.median(np.where(filtered[175:225] == 255)[0]) + 175

            self.scale.append([upper_scale, lower_scale])