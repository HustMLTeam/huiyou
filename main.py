# coding: utf-8
def show_im(mul, frame=None, *imgs):
    result = None
    height = imgs[0].shape[0]
    cut = cv2.cvtColor(np.ones((height, 1), dtype='uint8') * 255, cv2.COLOR_GRAY2BGR)
    for img in imgs:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if result is None:
            result = img
        else:
            result = np.hstack((result, cut, img))
    result = cv2.resize(result, None, fx=mul, fy=mul)

    if frame is not None:
        new_width = result.shape[1]
        new_height = int(frame.shape[0] / frame.shape[1] * new_width)
        new_frame = cv2.resize(frame, (new_width, new_height))
        result = np.vstack((new_frame, result))

    cv2.imshow('frame', result)

if __name__ == '__main__':
    import numpy as np
    import cv2
    from detect import Locator, Detector
    loc = None
    speed = 1
    count = 0
    cap = cv2.VideoCapture('data/tube1.mp4')
    while cap.isOpened():
        count += 1
        ret, frame = cap.read()
        if loc is None:
            loc = Locator(frame)
            # loc.locate_tube()
            # loc.locate_window()
            # loc.locate_scale()
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

            loc.tube = [(100, 583, 115, 208), (0, 657, 238, 331)]
            loc1 = [(124, 209, 143, 181),
                    (221, 304, 137, 177),
                    (314, 403, 137, 176),
                    (411, 487, 136, 181),
                    (500, 582, 143, 182)]

            loc2 = [(7, 92, 269, 308),
                    (94, 184, 261, 302),
                    (190, 284, 259, 302),
                    (288, 374, 260, 299),
                    (387, 477, 261, 303),
                    (481, 570, 266, 308),
                    (568, 651, 273, 314)]

            loc.window = [loc1, loc2]
            loc.scale = [[290, 526], [271, 525]]

            tube0 = Detector(loc.tube[0], loc.window[0], loc.scale[0], 350, 50)
            tube1 = Detector(loc.tube[1], loc.window[1], loc.scale[1], 450, 33)
        tube0.feed(frame)
        tube1.feed(frame)
        if not count % 20:
            tube0.update_backgrounds()
            tube1.update_backgrounds()

        level = int(tube0.cur_level)
        cv2.line(frame, (140, level), (180, level), (0, 0, 255), 1)
        level = int(tube1.cur_level)
        cv2.line(frame, (270, level), (310, level), (0, 0, 255), 1)
        text0 = 'level0: ' + str(round(tube0.level_scale))
        text1 = 'level1: ' + str(round(tube1.level_scale))
        cv2.putText(frame, text0, (5, 13), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6,
                    (0, 0, 255), 1)
        cv2.putText(frame, text1, (5, 28), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6,
                    (0, 0, 255), 1)

        # frame = cv2.resize(frame, None, fx=2, fy=2)
        # cv2.imshow('Result', frame)

        tube0.backgrounds
        show_im(1, None, )

        key = cv2.waitKey(speed) & 0xff
        if key == ord('q'):
            break
        elif key == ord('p'):
            cv2.waitKey()
        elif key == ord('c'):
            speed = 40 - speed
    cap.release()
    cv2.destroyAllWindows()