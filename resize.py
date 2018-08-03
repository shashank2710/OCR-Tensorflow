#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 15:39:38 2018

@author: ssatyanarayana
"""

from PIL import Image
import pytesseract

text = pytesseract.image_to_string(Image.open('Bottom Left_Chassis Number_0.jpg'))
