# coding: utf-8

import numpy as np


class SlideWindow(object):
    """
    SlideWindow(feature_extractor, classifier, decision, width_min=10,
               height_min=10, width_max=None, height_max=None, width_inc=3,
               height_inc=3, x_step=3, y_step=3, ratio_min=None, ratio_max=None)

        A slide window that can pick up eligible area from images.

        Parameters
        ----------
        feature_extractor : function
            A function that can extract features from an image.
        classifier : function
            A function that can classify image.
        decision : function
            A function that can decide windows from possible windows.
        width_min : int, optional (default=10)
            Minimum width of slide window.
        height_min : int, optional (default=10)
            Minimum height of slide window.
        width_max : int, optional
            Maximum width of slide window.
        height_max : int, optional
            Maximum height of slide window.
        width_inc : int, optional (default=3)
            Slide window's width will increase by width_inc.
        height_inc : int, optional (default=3)
            Slide window's height will increase by height_inc.
        x_step : int, optional (default=3)
            Slide window's x-coordinate will increase by x_step.
        y_step : int, optional (default=3)
            Slide window's y-coordinate will increase by y_step.
        ratio_min : float, optional
            Minimum value of height / width, it will decide slide window's
            minimum height.
            If None is given, window's minimum height will decide by height_min.
        ratio_max : float, optional
            Maximum value of height / width, it will decide slide window's
            maximum height.
            If None if given, window's maximum height will decide by height_max.
    """
    def __init__(self, feature_extractor, classifier, decision, width_min=10,
                 height_min=10, width_max=None, height_max=None, width_inc=3,
                 height_inc=3, x_step=3, y_step=3, ratio_min=None, ratio_max=None):
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.decision = decision
        self.w_min = width_min
        self.h_min = height_min
        self.w_max = width_max
        self.h_max = height_max
        self.w_inc = width_inc
        self.h_inc = height_inc
        self.x_step = x_step
        self.y_step = y_step
        self.r_min = ratio_min
        self.r_max = ratio_max

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

        w_min = self.w_min
        w_max = self.w_max if self.w_max and self.w_max < img_width else img_width
        w_inc = self.w_inc
        for w in range(w_min, w_max, w_inc):
            # find minimum window height
            if self.r_min:
                h_min = int(w * self.r_min)
                h_min = self.h_min if self.h_min > h_min else h_min
            else:
                h_min = self.h_min

            # find maximum window height
            h_max = [img_height]
            if self.r_max:
                h_max.append(int(w * self.r_max))
            if self.h_max:
                h_max.append(self.h_max)
            h_max = min(h_max)

            h_inc = self.h_inc

            for h in range(h_min, h_max, h_inc):
                for x_pos in range(0, img_width-w, self.x_step):
                    for y_pos in range(0, img_height-h, self.y_step):
                        window = image[y_pos:y_pos+h, x_pos:x_pos+w]
                        X = self.feature_extractor(window).reshape(1, -1)
                        if self.classifier(X)[0]:
                            windows = np.vstack((windows, (x_pos, y_pos,
                                                           x_pos+w, x_pos+h)))
        areas = self.decision(windows)
        return areas
