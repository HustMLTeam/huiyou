# coding: utf-8

from basic import FeatureExtractor, Classifier, Decision
from slide_window import slide_window

import numpy as np
import cv2


class LevelDetector(object):
    def __init__(self, tube_extract='sift', tube_classify='svm',
                 window_extract='lbp', window_classify='svm'):
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

    # 在图片上定位液位仪
    def locate_tube(self, image):
        positions = [] # 保存液位仪的位置，格式为(y_start, y_end, x_start, x_end)
        height, width = image.shape
        for y_start, y_end, x_start, x_end in slide_window(140, height,
                                width_min=22, width_max=32, width_inc=3,
                                height_min=200, height_max=240, height_inc=5,
                                ratio_min=7, ratio_max=9):
            feature = self.tube_extractor.extract(image[y_start:y_end, x_start:x_end])  # 提取特征
            if self.tube_classifier.classify(feature.reshape(1, -1)):  # 进行分类
                positions.append((y_start, y_end, x_start, x_end))
                # print('find position [%d:%d, %d:%d]' % (y_start, y_end, x_start, x_end))
        # 在大量找到的区域中选出合适的结果
        dec = Decision('max')
        return dec.decide(np.array(positions), 2)

    # 在液位仪上定位窗口
    def locate_window(self, tube, n_clusters, norm=False):
        positions = []
        height, width = tube.shape
        for y_start, y_end, x_start, x_end in slide_window(width, height,
                                width_min=13, width_max=25, width_inc=3,
                                height_min=26, height_max=39, height_inc=3,
                                x_step=2, y_step=2):
            img = tube[y_start:y_end, x_start:x_end]
            # 直方图均衡化
            if norm:
                img = cv2.equalizeHist(img)

            feature = self.window_extractor.extract(img)
            if self.window_classifier.classify(feature.reshape(1, -1)):
                positions.append((y_start, y_end, x_start, x_end))
                # print('find position [%d:%d, %d:%d]' % (y_start, y_end, x_start, x_end))
        dec = Decision('average')
        return dec.decide(np.array(positions), n_clusters)

    def locate_level(self):
        pass
