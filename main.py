# coding: utf-8

import numpy as np
import cv2
from detect import Locator, Detector


def main(file):
    loc = None
    speed = 1
    count = 0
    cap = cv2.VideoCapture(file)
    while cap.isOpened():
        count += 1
        ret, frame = cap.read()
        if loc is None:
            loc = Locator(frame, 12)
            loc.locate_window()
            loc.locate_scale()

            tube0 = Detector(loc.window[0], loc.scale[0])
            tube1 = Detector(loc.window[1], loc.scale[1])
        tube0.feed(frame)
        tube1.feed(frame)

        level = int(tube0.cur_level)
        x1 = max(window[2] for window in loc.window[0])
        x2 = min(window[3] for window in loc.window[0])
        cv2.line(frame, (x1, level), (x2, level), (0, 0, 255), 1)
        level = int(tube1.cur_level)
        x1 = max(window[2] for window in loc.window[1])
        x2 = min(window[3] for window in loc.window[1])
        cv2.line(frame, (x1, level), (x2, level), (0, 0, 255), 1)
        text0 = 'level0: ' + str(round(tube0.level_scale))
        text1 = 'level1: ' + str(round(tube1.level_scale))
        cv2.putText(frame, text0, (5, 13), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6,
                    (0, 0, 255), 1)
        cv2.putText(frame, text1, (5, 28), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6,
                    (0, 0, 255), 1)

        cv2.imshow('Result', frame)

        key = cv2.waitKey(speed) & 0xff
        if key == ord('q'):
            break
        elif key == ord('p'):
            cv2.waitKey()
        elif key == ord('c'):
            speed = 40 - speed
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main('data/tube1.mp4')