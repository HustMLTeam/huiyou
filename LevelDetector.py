# coding: utf-8

import numpy as np
from scipy.signal import convolve2d
import cv2
from collections import deque

from basic import horizontal_filter


class Detector(object):
    def __init__(self, tube_pos, window_pos, scale, init_level):
        self.tube_pos = tube_pos

        self.window_pos = []
        for pos in reversed(window_pos):
            y_start, y_end, x_start, x_end = pos
            if y_start < init_level < y_end:
                self.window_pos.append([init_level + 1, y_end, x_start, x_end])
                self.window_pos.append([y_start, init_level, x_start, x_end])
            else:
                self.window_pos.append(pos)

        self.upper_scale, self.lower_scale = scale
        self.init_level = init_level

        self.levels = deque([self.init_level] * 40, maxlen=40)

        self.frame = None
        self.first_avg = None
        self.cur_avg = None
        self.backgrounds = []
        self.p = 0.1
        self.t = 3
        self.d = 30

    @property
    def cur_level(self):
        return np.mean(self.levels)

    @property
    def level_scale(self):
        return 10 + 10 * (self.lower_scale - self.cur_level) / \
                    (self.lower_scale - self.upper_scale)

    def find_window(self, level):
        i = 0
        for i, (y_start, y_end, x_start, x_end) in enumerate(self.window_pos):
            if y_start < level < y_end:
                return i
            elif y_end <= level:
                continue
            elif y_start >= level:
                return i - 0.5
        return i + 0.5

    def feed(self, frame):
        if frame.ndim == 3:
            frame = frame[:, :, 0]

        self.cur_avg = frame[self.tube_pos[0]:self.tube_pos[1], self.tube_pos[2]:self.tube_pos[3]].mean()
        if self.first_avg is None:
            self.first_avg = self.cur_avg
            for y_start, y_end, x_start, x_end in self.window_pos:
                self.backgrounds.append(frame[y_start:y_end, x_start:x_end].astype('float64'))
            self.frame = frame.astype('float64')
        else:
            self.detect_level()

    def detect_level(self):
        result = []
        white = None
        for i, (y_start, y_end, x_start, x_end) in enumerate(self.window_pos):
            window = self.frame[y_start:y_end, x_start:x_end]
            window = window + self.first_avg - self.cur_avg
            foreground = abs(window - self.backgrounds[i])
            foreground[foreground <= self.t] = 0

            y1 = int(max(self.cur_level - self.d, y_start) - y_start)
            y2 = int(min(self.cur_level + self.d, y_end) - y_start)
            if y1 < y2 and white is None:
                if foreground.mean() > 15:
                    white = y2 + y_start
                else:
                    sobel = cv2.Sobel(foreground[y1:y2], cv2.CV_16S, 0, 1)
                    if y_start < self.init_level:
                        threshold = 33
                        sobel = np.where(sobel > threshold, 255, 0).astype('uint8')
                    else:
                        threshold = 25
                        sobel = np.where(sobel < -threshold, 255, 0).astype('uint8')
                    sobel_bl = horizontal_filter(sobel)
                    result += list(np.where(sobel_bl)[0] + y_start)

        if result:
            self.levels.append(np.median(result))
        elif white is not None:
            self.levels.append(white)

    def update_backgrounds(self):
        for i, (y_start, y_end, x_start, x_end) in enumerate(self.window_pos):
            window = self.frame[y_start:y_end, x_start:x_end]
            foreground = np.abs(self.backgrounds[i] - window)
            tmp = (1 - self.p) * self.backgrounds[i] + self.p * window
            self.backgrounds[i][foreground <= self.t] = tmp[foreground <= self.t]
