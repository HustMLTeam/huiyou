# coding: utf-8

import numpy as np
import cv2

if __name__ == '__main__':
    cap = cv2.VideoCapture('data/tube.mp4')
    while cap.isOpened():
