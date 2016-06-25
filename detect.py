# coding: utf-8

from slide_window import slide_window
from basic import FeatureExtractor, Classifier


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

        self.tube_extractor = FeatureExtractor('data/pkl/tube_%s.pkl' %
                                               (self.extract_method))
        self.window_extractor = FeatureExtractor('data/pkl/window_%s.pkl' %
                                                 (self.extract_method))
        self.tube_classifier = Classifier('data/pkl/tube_%s_%s.pkl' %
                                          (self.extract_method, self.classify_method))
        self.window_classifier = Classifier('data/pkl/window_%s_%s.pkl' %
                                            self.extract_method, self.classify_method)

    def locate_tube(self, image):
        height, width = image.shape
        for y_start, y_end, x_start, x_end in slide_window(width, height):
            feature = self.tube_extractor.extract(image[y_start:y_end, x_start:x_end])
            if self.tube_classifier.classify(feature):
                yield (y_start, y_end, x_start, x_end)

    def locate_window(self, tube):
        pass

    def locate_level(self):
        pass
