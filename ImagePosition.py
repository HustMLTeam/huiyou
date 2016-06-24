from classifier import Svm
from feature_extractor import Sift
from slide_window import SlideWindow
from decision import max_cover, average_cover
import cv2


def getTubePos(img):
    feature_extractor = Sift().tube()
    classifier = Svm().tube()
    decision = max_cover(2)
    s_win = SlideWindow(feature_extractor, classifier, decision, width_min=20,
                        width_max=45, width_inc=10, height_min=210,
                        height_max=250, height_inc=10, x_step=3, y_step=10)
    tubePos = s_win.slide(img)
    return tubePos


def getWindowPos(tube, n):
    feature_extractor = Sift().window()
    classifier = Svm().window()
    decision = average_cover(n)
    s_win = SlideWindow(feature_extractor, classifier, decision, width_min=13,
                        width_max=27, width_inc=3, height_min=25,
                        height_max=40, height_inc=3, x_step=3, y_step=4,
                        ratio_min=1.3, ratio_max=2.2)
    windowPos = s_win.slide(tube)
    return windowPos

if __name__=='__main__':
    print(getImgPos('data\\tube.jpg'))