import numpy as np
from scipy.signal import convolve2d
import cv2
from collections import deque


class LevelDetector(object):
    def __init__(self, tube_number, tube_pos, window_pos):
        assert tube_number in [1, 2]
        if tube_number == 1:
            self.init_level = 220
            self.init_window = -1
            self.cur_window = -1
            self.recg_distance = 20
            self.recg_size = 5
        if tube_number == 2:
            self.init_level = 160
            self.init_window = 2
            self.cur_window = 2
            self.recg_distance = 25
            self.recg_size = 2
            
        self.level_scale = 0
        self.upper_scale = 0
        self.lower_scale = 0
        self.cur_level = self.init_level
        self.levels = deque([self.init_level]*40, maxlen=40)
        self.first_avg = None
        self.backgrounds = []
        self.p = 0.1
        self.t = 3
        self.tube_pos = tube_pos
        self.window_pos = window_pos
        self.frame = None
        self.count = 0

    def detect_level(self, frame):
        frame_blue = frame[:, :, 0]
        tube = frame_blue[self.tube_pos[0]:self.tube_pos[1], self.tube_pos[2]:self.tube_pos[3]].astype('float64')
        lev_tmp = 0
        i_tmp = 0
        self.count += 1

        if self.count == 1:
            self.first_avg = tube.mean()
            for y_start, y_end, x_start, x_end in self.window_pos:
                self.backgrounds.append(frame_blue[y_start:y_end, x_start:x_end].astype('float64'))
            
            ruler = frame[self.tube_pos[0]:self.tube_pos[1], self.tube_pos[2]+15:self.tube_pos[3]+15]
            self.upper_scale, self.lower_scale = self.locate_scale(ruler)
            return self.cur_level, self.level_scale
        
        for i, (y_start, y_end, x_start, x_end) in enumerate(self.window_pos):
            window = frame_blue[y_start:y_end, x_start:x_end].astype('float64')
            window = window + self.first_avg - tube.mean()
            foreground = abs(window - self.backgrounds[i])
            foreground[foreground <= self.t] = 0

            if not self.count % 20 and window.mean() > self.backgrounds[i].mean():
                tmp = (1 - self.p) * self.backgrounds[i] + self.p * window
                self.backgrounds[i][foreground == 0] = tmp[foreground == 0]

            if abs(i - self.cur_window) > 1:
                continue

            sobel = cv2.Sobel(foreground, cv2.CV_16S, 0, 1)

            if i < self.init_window:
                sobel[sobel > 0] = 0
            if i > self.init_window:
                sobel[sobel < 0] = 0
            sobel = abs(sobel)
            
            if (window - self.backgrounds[i]).mean() > 15 and abs(self.first_avg - tube.mean()) < 3 and self.cur_level < self.init_level:
                self.cur_level = 210
                self.levels = deque([self.cur_level]*40, maxlen=40)

            ret, sobel_bi = cv2.threshold(sobel, 33, 255, cv2.THRESH_BINARY)
            sobel_bl = LevelDetector.median_filter(sobel_bi)

            tmp = 0
            if np.any(sobel_bl):
                area = np.array(np.where(sobel_bl == 255))
                area_size = area.shape[1]
                if area_size < self.recg_size:
                    continue
                if i == self.init_window:
                    lev_upper = y_start + area[0][0]
                    lev_lower = y_start + area[0][-1]
                    if abs(lev_upper - self.cur_level) < abs(lev_lower - self.cur_level):
                        tmp = lev_upper
                    else:
                        tmp = lev_lower
                else:
                    tmp = y_start + area[0][area_size//2]
            if abs(tmp - self.cur_level) < abs(lev_tmp - self.cur_level):
                lev_tmp = tmp
                i_tmp = i

        if abs(lev_tmp - self.cur_level) < self.recg_distance:
            self.levels.append(lev_tmp)
            self.cur_level = np.mean(self.levels)
            self.cur_window = i_tmp

        self.level_scale = 10 + 10 * (self.lower_scale - self.cur_level) / (self.lower_scale - self.upper_scale)
        return self.cur_level, self.level_scale

    def locate_scale(self, ruler):
        ruler = cv2.cvtColor(ruler, cv2.COLOR_BGR2GRAY)
        
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

        filtered = LevelDetector.median_filter(np.where(conv>15, 255, 0))
        upper_scale = np.median(np.where(filtered[75:125] == 255)[0]) + 75
        lower_scale = np.median(np.where(filtered[175:225] == 255)[0]) + 175

        return upper_scale, lower_scale

    @staticmethod
    def median_filter(src):
        result = convolve2d(src, np.ones((1, 5)) / 5, mode='same')
        result = np.where(result > 200, 255, 0).astype('uint8')
        return result
