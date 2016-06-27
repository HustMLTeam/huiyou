# coding: utf-8

import numpy as np
from slide_window import slide_window
from basic import FeatureExtractor, Classifier, Decision


class LevelDetector(object):
    def __init__(self, extract='sift', classify='svm'):
        # find feature extractor method and classify method
        if extract in ['sift', 'lbp']:
            self.extract_method = extract
        else:
            self.extract_method = 'sift'
        if classify in ['svm']:
            self.classify_method = 'svm'
        else:
            self.classify_method = 'svm'
        if self.extract_method == 'sift':
            self.tube_extractor = FeatureExtractor('sift', file='data/pkl/tube_sift.pkl')
            self.tube_classifier = Classifier(file='data/pkl/tube_sift_svm.pkl')
            self.window_extractor = FeatureExtractor('sift', file='data/pkl/window_sift.pkl')
            self.window_classifier = Classifier(file='data/pkl/window_sift_svm.pkl')
        elif self.extract_method == 'lbp':
            self.window_extractor = FeatureExtractor(method='lbp', file='data/pkl/window_lbp.pkl')
            self.window_classifier = Classifier(file='data/pkl/window_lbp_svm.pkl')

    def locate_tube(self, image):
        positions = []
        height, width = image.shape
        for y_start, y_end, x_start, x_end in slide_window(140, height,
                                width_min=22, width_max=32, width_inc=3,
                                height_min=200, height_max=240, height_inc=5,
                                ratio_min=7, ratio_max=9):
            feature = self.tube_extractor.extract(image[y_start:y_end, x_start:x_end])
            if self.tube_classifier.classify(feature.reshape(1, -1)):
                positions.append((y_start, y_end, x_start, x_end))
                print('find position [%d:%d, %d:%d]' % (y_start, y_end, x_start, x_end))
        dec = Decision('max')
        return dec.decide(np.array(positions), 2)

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
                img = ((img - img.min()) / (img.max() - img.min()) * 255).astype('uint8')
            feature = self.window_extractor.extract(img)
            if self.window_classifier.classify(feature.reshape(1, -1)):
                positions.append((y_start, y_end, x_start, x_end))
                # print('find position [%d:%d, %d:%d]' % (y_start, y_end, x_start, x_end))
        dec = Decision('average')
        return dec.decide(np.array(positions), n_clusters)

    def locate_level(self):
        pass
