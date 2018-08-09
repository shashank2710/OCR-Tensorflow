#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 15:39:38 2018

@author: ssatyanarayana
"""

import cv2

img=cv2.imread('ocr/3.jpg',0)

cv2.imshow('Original Image',img)

dim = (128, 128)

resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

(thresh, img_bw) = cv2.threshold(resized, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imshow('B&W Image',img_bw)

cv2.imwrite('ocr/resized.jpg', img_bw)

cv2.waitKey(0)
cv2.destroyAllWindows()