import sys
import os
import imghdr
import cv2

from PyQt4 import QtCore
from PyQt4 import QtGui
from PyQt4.QtGui import QMessageBox, QPixmap, QPainter, QPen, QImage

from threading import Thread
from time import sleep
#from numpy import reshape
import numpy as np
import MLS
import ImagePosition


#Machine Learning Multimedia System
class MLMS(QtGui.QMainWindow, MLS.Ui_MainWindow):
    def __init__(self, parent=None):
        super(self.__class__, self).__init__(parent)
        self.setupUi(self)
        self.actionImage.triggered.connect(self.openImage)
        self.actionVideo.triggered.connect(self.openVideo)
        self.actionExit.triggered.connect(self.exitApp)
        self.actionHelp.triggered.connect(self.showHelp)
        self.actionDrawBox_2.triggered.connect(self.startFindSomething)
        self.B_pause.clicked.connect(self.videoPause)
        self.B_next.clicked.connect(self.nextFrame)
        self.pixmap = None
        self.thread = None
        self.video_pause = False
        self.isDrawBox = False
        
    def showHelp(self):
        text = str(self.LW_feature.itemFromIndex(self.LW_feature.currentIndex()))+'\n'+str(dir(self.LW_feature.currentItem().text))
        self.Label_print.setText(text)
            
    def showErrMsg(self, text=None, infoText=None):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        if text:
            msg.setText(str(text))
        if infoText:
            msg.setInformativeText(str(infoText))
        msg.setWindowTitle('Error!')
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()
        
    def openImage(self):
        path = QtGui.QFileDialog.getOpenFileName(self, "Choose an Image")
        if path:
            try:
                assert imghdr.what(path), 'Not an image'
                self.pixmap = QPixmap(path)
                self.pixmap = self.pixmap.scaled(self.labelImage.size(),
                                                 QtCore.Qt.KeepAspectRatio)
                self.labelImage.setPixmap(self.pixmap)
            except Exception as e:
                text = "Can't open file {filename}".format(
                    filename = os.path.split(path)[1])
                info = str(e)
                self.showErrMsg(text, info)
                
    def openVideo(self):
        tips = "Choose an Video"
        expand = 'Image Files(*.*)'
        path = QtGui.QFileDialog.getOpenFileName(self, 
                    tips,QtGui.QDesktopServices.storageLocation(QtGui.QDesktopServices.MusicLocation), expand)
        
        self.videoCapture = cv2.VideoCapture(path)
        print(path)
        print(self.videoCapture)
        '''if self.videoCapture==None:没用
            return'''

        #获得码率及尺寸
        fps = self.videoCapture.get(cv2.CAP_PROP_FPS)
        size = (int(self.videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                int(self.videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        #指定写视频的格式, I420-avi, MJPG-mp4
        #videoWriter = cv2.VideoWriter('C:\\Users\\zying\\Desktop\\testVideo\\writed.avi', -1, fps, size)
        
        #视频的初始数据
        self.success, self.frame = self.videoCapture.read()
        self.waitTime = (1000/fps)
        self.VideoSlider.setMaximum(self.videoCapture.get(cv2.CAP_PROP_FRAME_COUNT) * self.waitTime/1000)
        #print(self.VideoSlider.maximum)
        self.video_time = 0.0
        
        #用线程播放视频
        self.createThread()
    
    def createThread(self):
        self.delThread()
        if not self.thread:
            #创建播放视频的线程
            self.thread=VideoThread()
            self.thread.con = self
            #注册信号处理函数
            self.thread._signal.connect(self.drawFrame)
        self.thread.waitTime = self.waitTime
        #启动线程
        self.thread.start()
        
    def restartThread(self):
        if self.thread:
            self.thread.start()
    
    def stopThread(self):
        if self.thread:
            self.thread.terminate()
            
    def delThread(self):
        if self.thread:
            self.thread.terminate()
    
    #播放一帧视频
    def drawFrame(self):
        if self.success:
            #cv2所得图像和qt的QImage图像的第一和第三通道不对应，应对调
            (r, g, b) = cv2.split(self.frame)
            img1 = cv2.merge([b,g,r])
            #将原来的（h,w,3）数组转换为（h*w,3）数组
            img = np.reshape(img1, (self.frame.shape[0]*self.frame.shape[1],3))
            #转换后的（h*w,3）数组可以转换为QImage
            #C++原代码为：QImage ( const uchar * data, int width, int height, Format format )
            QImg = QtGui.QImage(img, self.frame.shape[1],self.frame.shape[0],QtGui.QImage.Format_RGB888)
            #print(dir(QtGui.QImage))可以了解第四个参数的枚举常量
            
            self.pixmap = QPixmap(QImg)
            self.pixmap = self.pixmap.scaled(self.labelImage.size(),
                                          QtCore.Qt.KeepAspectRatio)
            self.labelImage.setPixmap(self.pixmap)
            self.success, self.frame = self.videoCapture.read() #获取下一帧
            
            #计算视频时间
            self.video_time += self.waitTime/1000
            #print(self.video_time)
            self.VideoSlider.setValue((int)(self.video_time))
            #self.VideoSlider.setSliderPosition(self.VideoSlider.value)
            self.T_second.setProperty("intValue", (int)(self.video_time%60))#避免显示60
            self.T_minute.setProperty("intValue", (int)(self.video_time%3600/60))
            self.T_hour.setProperty("intValue", (int)(self.video_time/3600))
            
            
            #画框框
            self.drawBoxPerFrame()
            
    def startFindSomething(self):
        if self.isDrawBox==False:
            self.isDrawBox = True
        else:
            self.isDrawBox = False
        
    def drawBoxPerFrame(self):
        if self.isDrawBox==False:
            return
        #获取当前帧图像并转化为灰度图
        img = cv2.cvtColor(self.frame,cv2.COLOR_BGR2GRAY) 
        self.pos = ImagePosition.getImgPos(img)
        #self.pos = np.array([[0,100,100,100],[200,100,100,200]])
        self.drawBox(self.pos[0])
        self.drawBox(self.pos[1])
        
            
                  
    def drawBox(self, rect):
        '''handle errors'''
        if not self.pixmap:
            self.showErrMsg('Please choose an image first!')
            return
        
        x, y, w, h = rect
        '''x, y, w, h = self.line_x.text(), self.line_y.text(), self.line_w.text(),\
            self.line_h.text()'''
        '''if not (x.isdigit() and y.isdigit() and w.isdigit() and h.isdigit()):
            self.showErrMsg('Please input integer!')
            return'''
        
        x, y, w, h = int(x), int(y), int(w), int(h)
        if not 0 <= x <= self.pixmap.width():
            self.showErrMsg('Value Error', 'x should be an integer between 0 and %d' 
                % self.pixmap.width())
        elif not 0 <= y <= self.pixmap.height():
            self.showErrMsg('Value Error', 'y should be an integer between 0 and %d'
                % self.pixmap.height())
        elif not 0 <= w <= (self.pixmap.width() - x):
            self.showErrMsg('Value Error', 'x+w should be an integer between 0 and %d'
                % self.pixmap.width())
        elif not 0 <= h <= (self.pixmap.height() - y):
            self.showErrMsg('Value Error', 'y+h should be an integer between 0 and %d'
                % self.pixmap.height())
        else:
            painter = QPainter()
            painter.begin(self.pixmap)
            #pen = QPen(QtCore.Qt.red, 5, QtCore.Qt.SolidLine)
            pen = QPen(QtCore.Qt.red, 1, QtCore.Qt.SolidLine)
            painter.setPen(pen)
            painter.drawRect(x, y, w, h)
            painter.end()
            self.labelImage.setPixmap(self.pixmap)
    
    def exitApp(self):
        self.stopThread()
        self.close()
        
    def videoPause(self):
        if self.video_pause==False:
            print('pause')
            self.video_pause = True
            self.stopThread()
        else:
            print('restart')
            self.video_pause = False
            #self.createThread()
            self.restartThread()
            
    def nextFrame(self):
        print('nextFrame')
        self.drawFrame()
        
        
class VideoThread(QtCore.QThread):
    _signal=QtCore.pyqtSignal()
    def __init__(self):
        super(VideoThread,self).__init__() 
        #定义信号,定义参数为str类型 
        self.waitTime = 1
        self.con = None
    def run(self):  
        while self.con.success:
            #发出信号
            self._signal.emit() 
            #让程序休眠
            #cv2.waitKey(self.waitTime)
            sleep(self.waitTime/1000)
            #sleep(0.04)        
            
class SliderThread(QtCore.QThread):
    _signal=QtCore.pyqtSignal(str)
    def __init__(self):
        super(VideoThread,self).__init__() 
        #定义信号,定义参数为str类型 
        self.imgPath = 'data\\tube.jpg'
        self.waitTime = 1
        self.con = None
    def run(self):  
        while self.con.success:
            #发出信号
            self._signal.emit(self.imgPath) 
            #让程序休眠
            #cv2.waitKey(self.waitTime)
            sleep(self.waitTime/1000)
            #sleep(0.04)

        
def main():
    app = QtGui.QApplication(sys.argv)
    form = MLMS()
    form.show()
    #exec_()线程事件循环
    sys.exit(app.exec_())
    

if __name__ == '__main__':
    main()
