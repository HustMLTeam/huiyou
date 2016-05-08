from classifier import Svm
from feature_extractor import Sift
from slide_window import SlideWindow
from decision import max_cover
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

def getWindowPos(tube): 
    feature_extractor = Sift().window()
    classifier = Svm().window()
    decision = max_cover(6)
    s_win = SlideWindow(feature_extractor, classifier, decision, width_min=10,
                        width_max=20, width_inc=1, height_min=30,
                        height_max=50, height_inc=2, x_step=2, y_step=4)
    windowPos = s_win.slide(tube)
    return windowPos

if __name__=='__main__':
    print(getImgPos('data\\tube.jpg'))