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
from collections import deque


class Locator(object):
    """
    **定位器**

    利用视频第一帧的图片找到两个液位仪的位置，第一个液位仪5个窗口的位置和第二个
    液位仪7个窗口的位置，以及两个液位仪刻度线的位置。

    定位的方法为利用滑动窗口，从图片中得到一个个小图片，然后利用特征提取器提取
    其特征向量，并利用分类器判断其是否符合条件。最后用决策器从所有符合条件的位
    置中得到最终的结果。

    使用时要用视频的第一帧图片实例化一个定位器，先调用loc_tube，再调用
    loc_window和loc_scale。例如：

      >>> loc = Locator(Image)
      >>> loc.locate_tube()
      >>> loc.locate_window()
      >>> loc.locate_scale()

    Parameters
    ----------
    image : 二维numpy数组
        视频第一帧的灰度图。

    tube_extract : str
        提取液位仪特征向量的方法。可以为'sift'或'lbp'，默认为'sift'。

    tube_classify : str
        对液位仪分类的方法。可以为'svm'，默认为'svm'。

    window_extract : str
        提取窗口特征的方法。可以为'sift'或'lbp'，默认为'lbp'。

    window_classify : str
        对窗口分类的方法。可以为'svm'，默认为'svm'。

    Attributes
    ----------
    tube : list
        两个液位仪的位置。第一个元素是第一个液位仪的位置，第二个元素是第二个液
        位仪的位置。

    window : list
        窗口的位置。第一个元素是第一个液位仪上的窗口的位置，第二个元素是第二个
        液位仪上的窗口的位置。

    scale : list
        刻度的位置。第一个元素是第一个液位仪两个刻度的位置，第二个元素是第二个
        液位仪两个刻度的位置。

    """
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

    def locate_tube(self):
        """
        定位液位仪。定位后的结果存放在tube属性中。
        """
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

    def locate_window(self):
        """
        定位窗口。定位的结果存放在window属性中。
        """
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
        """
        定位刻度。定位的结果存放在scale属性中。
        """
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



class Detector(object):
    """
    **液面检测器**
    
    利用背景差法进行液位检测。先取第一帧作为背景，以后每一帧减去背景就能得到前
    景，在前景上利用Sobel算子进行竖直方向上的梯度检测，当梯度的绝对值大于一定值
    时就可以认为它是边缘，在初始液面以上只需要保留梯度为正的边缘，在初始液面以
    下保留梯度为负的边缘，这个边缘就是液面的位置。为了减少噪声的干扰，还需要对
    得到的边缘进行一次水平滤波，只保留水平连续的一些点，舍弃掉孤立点。但这样仍
    然会存在噪声的干扰，于是采用目标跟踪的方法，只在上一帧液面的周围寻找液面，
    找到液面后不直接将其作为当前帧的液面，而是把它放入一个队列中，以这个队列的
    平均值作为当前帧的液面位置，以增强模型的稳定性。

    Parameters
    ----------
    tube_pos : list
        液位仪的位置。

    window_pos : list
        窗口的位置的列表。

    scale : list
        两条刻度线的位置，格式为(upper_scale, lower_scale)。

    init_level : int
        初始液面的位置。

    threshold : int
        用Sobel求梯度后，高于此阀值的将被保留，否则会被舍弃（置为0）。

    Attributes
    ----------
    cur_level : float
        当前液面所在像素的位置。

    level_scale : float
        当前液面的实际位置。
    """
    def __init__(self, tube_pos, window_pos, scale, init_level, threshold):
        self.tube_pos = tube_pos

        self.window_pos = []
        for y_start, y_end, x_start, x_end in reversed(window_pos):
            avg = int((y_start + y_end) / 2)
            y1 = avg - 12
            y2 = avg + 12
            avg = int((x_start + x_end) / 2)
            x1 = avg - 4
            x2 = avg + 4
            if y1 < init_level < y2:
                self.window_pos.append([init_level, y2, x1, x2])
                self.window_pos.append([y1, init_level, x1, x2])
            else:
                self.window_pos.append([y1, y2, x1, x2])

        self.upper_scale, self.lower_scale = scale
        self.init_level = init_level
        self.threshold = threshold

        self.levels = deque([self.init_level] * 40, maxlen=40)

        self.frame = None
        self.first_avg = None
        self.cur_avg = None
        self.backgrounds = []
        self.p = 0.1
        self.t = 3
        self.d = 25

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

        先将当前帧保留蓝色通道作为新的当前帧，在第一帧时初始化参数，随后的帧调
        用detect_level检测液面。

        Parameters
        ----------
        frame : 三维numpy数组
            当前帧的BGR图像。
        """
        if frame.ndim == 3:
            frame = frame[:, :, 0]

        self.frame = frame.astype('float64')
        self.cur_avg = frame[self.tube_pos[0]:self.tube_pos[1], self.tube_pos[2]:self.tube_pos[3]].mean()
        if self.first_avg is None:
            self.first_avg = self.cur_avg
            for y_start, y_end, x_start, x_end in self.window_pos:
                self.backgrounds.append(frame[y_start:y_end, x_start:x_end].astype('float64'))
        else:
            self.detect_level()

    def detect_level(self):
        """
        液面检测。

        由feed方法调用。
        """
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
                if foreground.mean() > 18:
                    white = y2 + y_start
                else:
                    sobel = cv2.Sobel(foreground[y1:y2], cv2.CV_16S, 0, 1)
                    if y_start < self.init_level:
                        threshold = max(self.backgrounds[i].std() * 2, self.threshold)
                        sobel = np.where(sobel > threshold, 255, 0).astype('uint8')
                    else:
                        threshold = max(self.backgrounds[i].std() * 2, 25)
                        sobel = np.where(sobel < -threshold, 255, 0).astype('uint8')
                    sobel_bl = horizontal_filter(sobel)
                    result += list(np.where(sobel_bl)[0] + y_start)

        if result:
            self.levels.append(np.median(result))
        elif white is not None:
            self.levels.append(white)

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
