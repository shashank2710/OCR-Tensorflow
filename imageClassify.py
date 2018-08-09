#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 12:14:00 2018

@author: ssatyanarayana
"""
#Import All the Required Parameters
from keras.models import load_model
import cv2
import numpy as np
import timeit
import sys

#Variables to return ASCII values and confidence
word =[]
confidence=[]
licensePlate=False	
dim = (128, 128)
    
start = timeit.timeit()

model = load_model('OCR_model.h5')

img=cv2.imread(sys.argv[1])

dataRef={"Sample01":'0', "Sample02":'1',"Sample03":'2',"Sample04":'3',"Sample05":'4'
         ,"Sample06":'5',"Sample07":'6',"Sample08":'7',"Sample09":'8',"Sample010":'9'
         ,"Sample011":'A',"Sample012":'B',"Sample013":'C',"Sample014":'D',"Sample015":'E'
         ,"Sample016":'F',"Sample017":'G',"Sample018":'H',"Sample019":'I',"Sample020":'J'
         ,"Sample021":'K',"Sample022":'L',"Sample023":'M',"Sample024":'N',"Sample025":'O'
         ,"Sample026":'P',"Sample027":'Q',"Sample028":'R',"Sample029":'S',"Sample030":'T'
         ,"Sample031":'U',"Sample032":'V',"Sample033":'W',"Sample034":'X',"Sample035":'Y',"Sample036":'Z'}

asciiRef={"Sample01":48, "Sample02":49,"Sample03":50,"Sample04":51,"Sample05":52
         ,"Sample06":53,"Sample07":54,"Sample08":55,"Sample09":56,"Sample010":57
         ,"Sample011":65,"Sample012":66,"Sample013":67,"Sample014":68,"Sample015":69
         ,"Sample016":70,"Sample017":71,"Sample018":72,"Sample019":73,"Sample020":74
         ,"Sample021":75,"Sample022":76,"Sample023":77,"Sample024":78,"Sample025":79
         ,"Sample026":80,"Sample027":81,"Sample028":82,"Sample029":83,"Sample030":84
         ,"Sample031":85,"Sample032":86,"Sample033":87,"Sample034":88,"Sample035":89,"Sample036":90}

if (licensePlate==True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,70,255,cv2.THRESH_BINARY_INV)
    cv2.imshow('Threshold Image', thresh)
    
    kernel = np.ones((1,1), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)
    cv2.imshow('Dialated Image', img_dilation)
    
    im2,cntrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    
    cv2.imshow('Contours Image', img)
    
    sorted_ctrs = sorted(cntrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
    
    for i, ctr in enumerate(sorted_ctrs):
        x, y, w, h = cv2.boundingRect(ctr)
        roi = img[y-5:y+h+5, x-5:x+w+5]
        
        if 50> w >2 and 50> h > 5:
            resized = cv2.resize(roi, dim, interpolation = cv2.INTER_AREA)
            thresh, img_bw = cv2.threshold(resized, 70, 255, cv2.THRESH_BINARY)
            cv2.imwrite('ocr/{}.jpg'.format(i), img_bw)
            x = np.expand_dims(img_bw, axis=0)
            detections = model.predict(x)
            word.append(dataRef['Sample0{}'.format(detections.argmax()+1)])
            confidence.append(np.max(detections))

else:
    #resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    thresh, img_bw = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY)
    x = np.expand_dims(img_bw, axis=0)
    detections = model.predict(x)
    word.append(dataRef['Sample0{}'.format(detections.argmax()+1)])
    word.append(asciiRef['Sample0{}'.format(detections.argmax()+1)])
    confidence.append(np.max(detections))
    
print(word,np.mean(confidence))
end = timeit.timeit()
print("Time Taken=",end-start)
