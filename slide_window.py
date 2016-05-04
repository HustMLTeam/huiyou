# coding: utf-8

import numpy as np


class SlideWinodw(object):
    """
    SlideWindow(feature_extractor, classifier, decision, width_min=10,
                 height_min=10, width_inc=3, height_inc=3, x_step=3, y_step=3,
                 radio_min=None, radio_max=None)

        A slide window that can pick up eligible area from images.

        Parameters
        ----------
        feature_extractor : function
            A function that can extract features fram an image.
        classifier : function
            A function that can classify image.
        decision : function
            A function that can deside windows from possible windows.
        width_min : int, optional (default=10)
            Minimum width of windows.
        height_min : int, optional (default=10)
            Minimum height of windows.
        width_inc : int, optional (default=3)
            Slide window's width will increase by width_inc.
        height_inc : int, optional (default=3)
            Slide window's height will increase by height_inc.
        x_step : int, optional (default=3)
            Slide window's x-coordinate will increase by x_step.
        y_step : int, optional (default=3)
            Slide window's y-coordinate will increase by y_step.
        radio_min : float, optional
            Minimum value of height / width, it will deside slide window's
            minimum height.
            If None is given, window's minimum height will deside by height_min.
        radio_max : float, optional
            Maximum value of height / width, it will deside slide window's
            maximum height.
            If None if given, window's minimum height will deside by height_min.
    """
    def __init__(self, feature_extractor, classifier, decision, width_min=10,
                 height_min=10, width_inc=3, height_inc=3, x_step=3, y_step=3,
                 radio_min=None, radio_max=None):
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.decision = decision
        self.w_min = width_min
        self.h_min = height_min
        self.w_inc = width_inc
        self.h_inc = height_inc
        self.x_step = x_step
        self.y_step = y_step
        self.r_min = radio_min
        self.r_max = radio_max

    def slide(self, image):
        """
        Pick up eligible areas from an image.

        Parameters
        ----------
        image: ndarray

        Returns
        -------
        areas : ndarray
        """
        assert isinstance(image, np.ndarray), 'Image should be an instance of \
            numpy.ndarray'
        img_height, img_width = image.shape
        windows = np.array([], dtype=np.int64).reshape(0, 4)
        for w in range(self.w_min, img_width, self.w_inc):
            # find minimum window height
            if self.r_min:
                h_min = int(w * self.r_min)
                h_min = self.h_min if self.h_min > h_min else h_min
            else:
                h_min = self.h_min
            # find maximum window height
            if self.r_max:
                h_max = int(w * self.r_max)
                h_max = img_height if img_height < h_max else h_max
            else:
                h_max = img_height

            for h in range(h_min, h_max, self.h_inc):
                for x_pos in range(0, img_width-w, self.x_step):
                    for y_pos in range(0, img_height-h, self.y_step):
                        window = image[y_pos:y_pos+h, x_pos:x_pos+w]
                        X = self.feature_extractor(window).reshape(1, -1)
                        if self.classifier(X)[0]:
                            windows = np.vstack((windows, (x_pos, y_pos, w, h)))
        areas = self.decision(windows)
        return areas
