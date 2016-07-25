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
            loc.locate_tube()
            loc.locate_window()
            loc.locate_scale()
            # loc.tube = [(0, 271, 33, 76), (0, 271, 78, 115)]
            # loc.window = [[(47, 80, 43, 61),
            #               (82, 115, 42, 58),
            #               (117, 151, 39, 58),
            #               (153, 186, 41, 61),
            #               (188, 220, 43, 61)],
            #              [(2, 35, 88, 108),
            #               (35, 69, 85, 105),
            #               (72, 105, 84, 105),
            #               (108, 142, 84, 104),
            #               (145, 178, 84, 104),
            #               (179, 213, 87, 108),
            #               (214, 246, 89, 109)]]
            # loc.scale = [[110, 199], [102, 199]]

            tube0 = Detector(loc.tube[0], loc.window[0], loc.scale[0], 210, 50)
            tube1 = Detector(loc.tube[1], loc.window[1], loc.scale[1], 160, 33)
        tube0.feed(frame)
        tube1.feed(frame)
        if not count % 20:
            tube0.update_backgrounds()
            tube1.update_backgrounds()

        if count < 5000:
            continue

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