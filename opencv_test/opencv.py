#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@version: python3.7
@author: ‘changzhaoliang‘
@license: Apache Licence 
@file: opencv.py
@time: 2019-04-17 13:23
"""

import cv2

import numpy as np
import matplotlib.pyplot as plt

# read image
BLUE = [0, 0, 255]
img = cv2.imread('../image/test.jpg')
# hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
# ret,dst = cv2.threshold(img,200,255,cv2.THRESH_BINARY)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


# laplacian = cv2.Laplacian(img, cv2.CV_32F)

# gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
# 边缘检测
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

lines = cv2.HoughLines(edges, 1, np.pi/90, 300)
for line in lines:
    for rho, theta in line:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 3000 * (-b))
        y1 = int(y0 + 3000 * (a))
        x2 = int(x0 - 3000 * (-b))
        y2 = int(y0 - 3000 * (a))
        cv2.line(gray, (x1, y1), (x2, y2), (0, 0, 255), 2)


cv2.imwrite('./image/test2.jpg', gray)

# img2 = cv2.imread('./image/test2.jpg')
# kernel = np.ones((3,3),np.uint8)
# dilation = cv2.dilate(img2,kernel,iterations = 1)


# cv2.namedWindow("Image", 2)
# cv2.imshow("Image", dilation)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
