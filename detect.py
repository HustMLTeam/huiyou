# coding: utf-8
"""
detect
======

该模块为功能类模块，提供液位仪定位与液面检测功能。
"""

from basic import FeatureExtractor, Classifier, Decision, horizontal_filter
from slide_window import slide_window

import numpy as np
from scipy.signal import convolve2d
import cv2
from collections import deque, Counter


class Locator(object):
    """
    **定位器**

    利用视频第一帧的图片找到各个窗口的位置，以及两个液位仪刻度线的位置。

    定位的方法为利用滑动窗口，从图片中得到一个个小图片，然后利用特征提取器提取
    其特征向量，并利用分类器判断其是否符合条件。最后用决策器从所有符合条件的位
    置中得到最终的结果。

    使用时要用视频的第一帧图片实例化一个定位器，先调用loc_window，再调用
    loc_scale。例如：

      >>> loc = Locator(Image, 12)
      >>> loc.locate_window()
      >>> loc.locate_scale()

    Parameters
    ----------
    image : 二维numpy数组
        视频第一帧的灰度图。

    total : int
        窗口的个数

    Attributes
    ----------
    window : list
        窗口的位置。第一个元素是第一个液位仪上的窗口的位置，第二个元素是第二个
        液位仪上的窗口的位置。

    scale : list
        刻度的位置。第一个元素是第一个液位仪两个刻度的位置，第二个元素是第二个
        液位仪两个刻度的位置。

    """
    def __init__(self, image, total):
        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) \
            if image.ndim == 3 else image
        self.total = total

        # 加载相应的特征提取器及分类器
        self.extractor = FeatureExtractor(file='data/pkl/window_lbp.pkl')
        self.classifier = Classifier(file='data/pkl/window_lbp_svm.pkl')

        self.window = [[], []]
        self.scale = []

    def locate_window(self):
        """
        定位窗口。定位的结果存放在window属性中。
        """
        positions = []
        height, width = self.image.shape
        for y_start, y_end, x_start, x_end in slide_window(width, height,
                            # width_min=35, width_max=45, width_inc=3,
                            # height_min=70, height_max=100, height_inc=3,
                            # x_step=2, y_step=2
                            ratio_min=1.3, ratio_max=2.3):
            img = self.image[y_start:y_end, x_start:x_end]
            feature = self.extractor(img)
            if self.classifier(feature.reshape(1, -1)):
                positions.append((y_start, y_end, x_start, x_end))
        dec = Decision('average')
        result = dec.decide(np.array(positions), self.total)
        result.sort(key=lambda t: t[2])
        mid = (result[0][2] + result[-1][2]) / 2
        for pos in result:
            if pos[2] < mid:
                self.window[0].append(pos)
            else:
                self.window[1].append(pos)
        self.window[0].sort(key=lambda t: t[0])
        self.window[1].sort(key=lambda t: t[0])
        self.window[0] = self.relocate_window(self.window[0])
        self.window[1] = self.relocate_window(self.window[1])

    def relocate_window(self, window):
        """
        重新定位窗口。

        利用滑动窗口对各个窗口进行定位，得到的窗口范围会比较大，位置仍然会有些偏差，这时对窗口进行重新定位，以得到更加准确的结果。

        Parameters
        ----------
        window : list
            初步定位得到的窗口位置。

        Returns
        -------
        list
            精确定位得到的窗口位置。
        """
        window_num = len(window)
        avg_dist = int((sum(window[-1][:2]) - sum(window[0][:2])) / (2 * (window_num - 1)))
        # 重新设定窗口宽度
        width = int(avg_dist * 0.15)
        window = [(y1, y2, int((x1+x2)/2-width), int((x1+x2)/2+width))
                  for y1, y2, x1, x2 in window]

        # 寻找窗口的上边缘
        sobel = np.zeros(self.image.shape)
        for y1, y2, x1, x2 in window:
            tmp = cv2.Sobel(self.image[y1:y2, x1:x2], cv2.CV_16S, 0, 1)
            tmp = np.where(tmp < -50, 255, 0).astype('uint8')
            tmp = horizontal_filter(tmp, width//2, width//2)
            sobel[y1:y2, x1:x2] = tmp
        points = np.where(sobel)[0]
        y_start = [None] * window_num
        i = window_num
        for y1, y2, x1, x2 in reversed(window):
            i -= 1
            y2 = int((y2 - y1) / 3 + y1)
            l = [y for y in points if y1 < y < y2]
            if l and (i == window_num - 1 or y_start[i+1] is None):
                y_start[i] = max(set(l), key=l.count)
            elif i < window_num - 1 and y_start[i+1] is not None:
                mid = y_start[i+1] - avg_dist
                for gap in range(3, 15, 3):
                    search = [y for y in l if mid - gap < y < mid + gap]
                    if search:
                        y_start[i] = max(set(search), key=search.count)
                        break
                y_start[i] = y_start[i+1] - avg_dist
        for i in range(window_num-1):
            if y_start[i+1] is None:
                y_start[i+1] = y_start[i] + avg_dist
        y_start = [y+3 for y in y_start]

        # 确定窗口的下边缘
        y_end = []
        for i in range(window_num-1):
            y_end.append(int((y_start[i+1] - y_start[i]) * 2 / 3 + y_start[i]))
        y_end.append(y_end[i] + avg_dist)
        y_end = [y-3 for y in y_end]
        return [(y_start[i], y_end[i], window[i][2], window[i][3]) for i in range(window_num)]

    def locate_scale(self):
        """
        定位刻度。定位的结果存放在scale属性中。
        """
        for window in self.window:
            window_num = len(window)
            dist = (sum(window[-1][:2]) - sum(window[0][:2])) / (2 * (window_num - 1))
            y1 = window[0][0]
            y2 = window[-1][1]
            x1 = max(w[3] for w in window)
            x2 = int(x1 + dist / 2)
            ruler = self.image[y1:y2, x1:x2]

            t = int(dist / 4)
            conv = convolve2d(ruler, np.array([[1], [1], [0], [-4], [0], [1], [1]]).
                              repeat(t, axis=1) / (8 * t), mode='same') - \
                   np.abs(convolve2d(ruler, np.array([[1], [1], [0], [0], [0],
                                                      [-1], [-1]]).
                                     repeat(8, axis=1) / (4 * t), mode='same'))
            leng = int(dist / 8.5)
            filtered = horizontal_filter(np.where(conv > 15, 255, 0), leng, leng)
            c = Counter(np.where(filtered)[0])
            high, low = sorted(d[0] for d in c.most_common(2))
            if low - high > 2 * dist:
                upper_scale = high + y1
                lower_scale = low + y1
            else:
                upper_scale = high + y1
                lower_scale = (low - high) * 2 + high + y1
            self.scale.append([upper_scale, lower_scale])



class Detector(object):
    """
    **液面检测器**

    分别在蓝色通道图和前景上进行液位检测。

    在蓝色通道图上进行液位检测时，对每个窗口求竖直梯度，并且只保留由上到下从白
    色过度到黑色的梯度，如果发现较长的边缘，则该边缘为当前液面。

    利用背景差法进行液位检测时，先取第一帧作为背景，以后每一帧减去背景就能得到前
    景，在前景上利用Sobel算子进行竖直方向上的梯度检测，当梯度的绝对值大于一定值
    时就可以认为它是边缘，在初始液面以上只需要保留梯度为正的边缘，在初始液面以
    下保留梯度为负的边缘，这个边缘就是液面的位置。为了减少噪声的干扰，还需要对
    得到的边缘进行一次水平滤波，只保留水平连续的一些点，舍弃掉孤立点。但这样仍
    然会存在噪声的干扰，于是采用目标跟踪的方法，只在上一帧液面的周围寻找液面，
    找到液面后不直接将其作为当前帧的液面，而是把它放入一个队列中，以这个队列的
    平均值作为当前帧的液面位置，以增强模型的稳定性。

    优先在蓝色通道上寻找液位，如果没有找到，再利用背景差法进行液位检测。

    Parameters
    ----------
    window_pos : list
        窗口的位置的列表。

    scale : list
        两条刻度线的位置，格式为(upper_scale, lower_scale)。

    init_level : int，可选
        初始液面的位置。缺省为最下面窗口的下边缘。

    threshold : int，可选
        用Sobel求梯度后，高于此阀值的将被保留，否则会被舍弃（置为0）。缺省为30.

    Attributes
    ----------
    cur_level : float
        当前液面所在像素的位置。

    level_scale : float
        当前液面的实际位置。
    """
    def __init__(self, window_pos, scale, init_level=None, threshold=None):
        self.window_pos = window_pos
        self.window_num = len(window_pos)
        self.avg_dist = (sum(window_pos[-1][:2]) - sum(window_pos[0][:2])) / (self.window_num - 1) / 2

        self.upper_scale, self.lower_scale = scale
        self.init_level = init_level if init_level else window_pos[-1][1]
        self.threshold = threshold if threshold else 30

        self.levels = deque([self.init_level] * 40, maxlen=40)

        self.frame = None
        self.windows = [None] * self.window_num
        self.backgrounds = None
        self.p = 0.1
        self.t = 3
        self.d = int(self.avg_dist * 0.3)
        self.count = 0

    @property
    def cur_level(self):
        return np.mean(self.levels)

    @property
    def level_scale(self):
        return 10 + 10 * (self.lower_scale - self.cur_level) / \
                    (self.lower_scale - self.upper_scale)

    def feed(self, frame):
        """
        将当前帧传入液面检测器。

        先调用detect_blue方法，如果找不到，再调用detect_background方法。

        Parameters
        ----------
        frame : 三维numpy数组
            当前帧的BGR图像。
        """
        # 将当前帧转化为蓝色通道图
        if frame.ndim == 3:
            self.frame = frame[:, :, 0].astype('float64')
        elif frame.ndim == 2:
            self.frame = frame.astype('float64')
        height, width = self.frame.shape
        y = int(height * 0.88888888888)
        x = int(width * 0.44444444444)
        self.frame[y:, :x] = 255

        for i, (y1, y2, x1, x2) in enumerate(self.window_pos):
            self.windows[i] = self.frame[y1:y2, x1:x2]
        level = self.detect_blue()
        if level:   #先在蓝色通道图上找液面
            self.levels.extend([level]*40)
            self.init_level = level
            self.backgrounds = None
            self.count = 0
        else:       #如果找不到，在前景图上找液面
            self.count += 1
            if not self.count % 20:     #每20帧更新背景
                self.update_backgrounds()
            level = self.detect_foreground()
            if level:
                self.levels.append(level)

    def detect_blue(self):
        """
        利用蓝色通道图检测当前液面。

        由feed方法调用。
        """
        i = self.window_num
        for window in reversed(self.windows[1:]):
            i -= 1
            w_bl = cv2.medianBlur(np.uint8(window), 5)
            w_edge = cv2.Sobel(w_bl, cv2.CV_16S, 0, 1)
            w_edge = np.where(w_edge < - 50, 255, 0)
            length = int(self.avg_dist / 8)
            w_edge_fil = horizontal_filter(w_edge, length, length)
            if np.any(w_edge_fil):
                return int(np.median(np.where(w_edge_fil)[0])) + self.window_pos[i][0]
        return None

    def detect_foreground(self):
        """
        利用背景差法寻找当前液面。

        由feed方法调用。
        """
        # 如果没有背景图，添加背景，并开始计数
        if self.backgrounds is None:
            print('init')
            self.backgrounds = self.windows[:]
            self.count = 1

        # 在所有窗口前景中寻找边缘
        sobel = []      # 保存所有窗口中符合条件的点的纵坐标
        for i in range(self.window_num):
            fore = np.abs(self.windows[i] - self.backgrounds[i]).astype('uint8')
            fore = cv2.medianBlur(fore, 3)
            y1, y2, x1, x2 = self.window_pos[i]
            y1 += 2
            y2 -= 2
            s = cv2.Sobel(fore[2:-2], cv2.CV_16S, 0, 1)
            if y2 <= self.init_level:       # 如果是在初始线以上，只寻找大于0的梯度
                s = np.where(s > self.threshold, 255, 0)
                s = horizontal_filter(s, 6, 6)
                sobel.extend(y + y1 for y in np.where(s)[0])
            elif y1 >= self.init_level:     # 如果是在初始线以下，只寻找小于0的梯度
                s = np.where(s < -self.threshold, 255, 0)
                s = horizontal_filter(s, 6, 6)
                sobel.extend(y + y1 for y in np.where(s)[0])
            else:
                # 在初始线以上寻找大于0的梯度
                up = s[:self.init_level-y1-1]
                up = np.where(up > self.threshold, 255, 0)
                up = horizontal_filter(up, 6, 6)
                sobel.extend(y + y1 for y in np.where(up)[0])
                # 在初始线以下寻找小于0的梯度
                down = s[self.init_level-y1+1:]
                down = np.where(down < -self.threshold, 255, 0)
                down = horizontal_filter(down, 6, 6)
                sobel.extend(y + y1 for y in np.where(down)[0])

        # 找到当前液位所在的窗口
        cur_window = self.window_num - 1
        for i, (y1, y2, x1, x2) in enumerate(self.window_pos):
            if self.cur_level <= y2:
                cur_window = i
                break

        # 缩小范围
        up = self.cur_level - self.d
        down = self.cur_level + self.d
        if up < self.window_pos[cur_window][0] and cur_window != 0:
            up -= self.window_pos[cur_window][0] - self.window_pos[cur_window-1][1]
        if down > self.window_pos[cur_window][1] and cur_window != self.window_num-1:
            down += self.window_pos[cur_window+1][0] - self.window_pos[cur_window][0]

        # 确定最终留下来的点
        select = [y for y in sobel if up <= y <= down]

        if select:
            # 选取中位数作为最终的结果
            return int(np.median(select))
        else:
            return None

    def update_backgrounds(self):
        """
        更新背景。

        由于环境会不断变化，例如光照，因此如果只拿第一帧作为背景图片的话时间一
        长就可能不准确，因此要不断地更新背景。更新背景不必每帧都进行，可以隔几
        帧更新一次，例如每20帧更新一次。
        """
        for i, (y_start, y_end, x_start, x_end) in enumerate(self.window_pos):
            window = self.frame[y_start:y_end, x_start:x_end]
            foreground = np.abs(self.backgrounds[i] - window)
            tmp = (1 - self.p) * self.backgrounds[i] + self.p * window
            self.backgrounds[i][foreground <= self.t] = tmp[foreground <= self.t]
