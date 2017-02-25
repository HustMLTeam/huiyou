# -*- coding: utf-8 -*-
"""
该模块用于从视频中取得训练集。
"""

import numpy as np
import cv2
import os
from random import randint


def getPos():
    position = [(46, 45, 62, 77),
           (43, 83, 60, 113),
           (43, 119, 59, 148),
           (43, 154, 60, 184),
           (45, 189, 62, 219),
           (90, 2, 108, 34),
           (88, 36, 105, 70),
           (85, 73, 105, 107),
           (85, 108, 106, 144),
           (86, 145, 107, 180),
           (87, 180, 109, 215),
           (90, 216, 110, 245)]

    count = 0
    cap = cv2.VideoCapture('./data/tube.mp4')
    dire = './data/window/pos/'
    
    while cap.isOpened():
        ret, frame = cap.read()
        count += 1
        if not count % 600:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for i, pos in enumerate(position):
                filename = os.path.join(dire, '%02d-%02d.jpg' % (count/600, i))
                cv2.imwrite(filename, gray[pos[1]:pos[3], pos[0]:pos[2]]) 
        
    cap.release()
    

def getNeg():
    cap = cv2.VideoCapture('./data/tube.mp4')
    dire = './data/window/neg/'
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        count += 1
        if not count % 30:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            im_height, im_width = gray.shape
            height = randint(21, 27)+randint(0, 5)
            width = randint(9, 11)+randint(0, 2)
            x = randint(0, im_height - height - 1)
            y = randint(0, im_width - width - 1)
            filename = os.path.join(dire, '%04d.jpg' % (count / 30))
            cv2.imwrite(filename, gray[x:x+height, y:y+width])
            
    cap.release()
    

if __name__ == '__main__':
    getPos()
