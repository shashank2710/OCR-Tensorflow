#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 13:47:20 2018

@author: ssatyanarayana
"""

import cv2
import numpy as np


img=cv2.imread('ocr/Top Center_Container Number_0.jpg')
if img is not None:
    cv2.imshow('Original Image',img)
    
else:
    print('Error Loading Image')

dim = (128, 128)
 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gray,70,255,cv2.THRESH_BINARY_INV)
cv2.imshow('Threshold Image', thresh)

kernel = np.ones((1,1), np.uint8)
img_dilation = cv2.dilate(thresh, kernel, iterations=1)
cv2.imshow('Dialated Image', img_dilation)

im2,cntrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#cv2.drawContours(img, cntrs, -1, (0,0,0), 3)
cv2.imshow('Contours Image', img)

sorted_ctrs = sorted(cntrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

for i, ctr in enumerate(sorted_ctrs):
    x, y, w, h = cv2.boundingRect(ctr)
 
    roi = img[y-5:y+h+5, x-5:x+w+5]
 
    #cv2.rectangle(image,(x,y),( (x+5) + (w+5), (y+5) + (h+5) ),(255,0,0),2)
 
    if 50> w >2 and 50> h > 5:
        resized = cv2.resize(roi, dim, interpolation = cv2.INTER_AREA)
        thresh, img_bw = cv2.threshold(resized, 70, 255, cv2.THRESH_BINARY)
        cv2.imwrite('ocr/{}.jpg'.format(i), img_bw)
cv2.waitKey(0)
cv2.destroyAllWindows()