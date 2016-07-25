# coding: utf-8


if __name__ == '__main__':
    import numpy as np
    import cv2
    from detect import Locator
    from LevelDetector import Detector
    loc = None
    speed = 1
    count = 0
    cap = cv2.VideoCapture('data/tube.mp4')
    while cap.isOpened():
        count += 1
        ret, frame = cap.read()
        if loc is None:
            loc = Locator(frame)
            loc.tube = [[0, 271, 33, 70],
                        [0, 271, 78, 115]]
            loc.window = [[[49, 77, 48, 61],
                           [85, 112, 46, 58],
                           [121, 148, 45, 56],
                           [159, 183, 47, 56],
                           [192, 216, 47, 59]],
                          [[5, 33, 92, 105],
                           [40, 68, 90, 103],
                           [76, 105, 89, 101],
                           [113, 141, 89, 101],
                           [150, 177, 90, 102],
                           [185, 210, 92, 104],
                           [219, 244, 95, 107]]]
            loc.scale = [[110, 200], [103, 200]]

            tube0 = Detector(loc.tube[0], loc.window[0], loc.scale[0], 220, 50)
            tube1 = Detector(loc.tube[1], loc.window[1], loc.scale[1], 160, 33)
        tube0.feed(frame)
        tube1.feed(frame)
        if not count % 20:
            tube0.update_backgrounds()
            tube1.update_backgrounds()

        level = int(tube0.cur_level)
        cv2.line(frame, (45, level), (60, level), (0, 0, 255), 1)
        level = int(tube1.cur_level)
        cv2.line(frame, (90, level), (105, level), (0, 0, 255), 1)
        text0 = 'level0: ' + str(round(tube0.level_scale))
        text1 = 'level1: ' + str(round(tube1.level_scale))
        cv2.putText(frame, text0, (5, 13), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6,
                    (0, 0, 255), 1)
        cv2.putText(frame, text1, (5, 28), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6,
                    (0, 0, 255), 1)

        frame = cv2.resize(frame, None, fx=2, fy=2)
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