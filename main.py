# coding: utf-8

import numpy as np
import cv2
from detect import Locator, Detector


def main(file):
    loc = None
    speed = 1   # 播放速度
    count = 0
    # 播放视频
    cap = cv2.VideoCapture(file)
    while cap.isOpened():
        count += 1
        ret, frame = cap.read()
        if loc is None: # 第一帧确定液位仪窗口位置以及刻度位置
            loc = Locator(frame, 12)
            loc.locate_window()
            loc.locate_scale()
            # 初始化两个液位探测器
            tube0 = Detector(loc.window[0], loc.scale[0])
            tube1 = Detector(loc.window[1], loc.scale[1])
        # 将当前画面传给两个液位探测器
        tube0.feed(frame)
        tube1.feed(frame)
        
        # 在画面上显示液面位置
        level = int(tube0.cur_level)
        x1 = max(window[2] for window in loc.window[0])
        x2 = min(window[3] for window in loc.window[0])
        cv2.line(frame, (x1, level), (x2, level), (0, 0, 255), 1)   # 显示左边液位仪的液位线
        level = int(tube1.cur_level)
        x1 = max(window[2] for window in loc.window[1])
        x2 = min(window[3] for window in loc.window[1])
        cv2.line(frame, (x1, level), (x2, level), (0, 0, 255), 1)   # 显示左边液位仪的液位线
        # 显示文字
        text0 = 'level0: ' + str(round(tube0.level_scale))
        text1 = 'level1: ' + str(round(tube1.level_scale))
        cv2.putText(frame, text0, (5, 13), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6,
                    (0, 0, 255), 1)
        cv2.putText(frame, text1, (5, 28), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6,
                    (0, 0, 255), 1)

        cv2.imshow('Result', frame)

        key = cv2.waitKey(speed) & 0xff
        if key == ord('q'): # 退出
            break
        elif key == ord('p'):   # 暂停
            cv2.waitKey()
        elif key == ord('c'):   # 改变播放速度
            speed = 40 - speed
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main('data/tube1.mp4')