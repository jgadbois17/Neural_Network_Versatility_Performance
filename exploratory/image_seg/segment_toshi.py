#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Segmentation 
Image: Picture of Toshi 
"""

import numpy as np 
import matplotlib.pyplot as plt 
from skimage import img_as_float, segmentation, filters, color 

#%%

# load and plot image 
image = plt.imread('exploratory/image_seg/toshi.jpg') 
image = color.rgb2gray(img_as_float(image)) 

plt.figure(figsize=(7, 7)) 
plt.imshow(image, cmap='gray')  
plt.show() 

#%%

thresholds = filters.threshold_multiotsu(image, classes=3) 
regions = np.digitize(image, bins=thresholds) 

plt.figure(figsize=(7, 7)) 
plt.imshow(regions, cmap='gray')  
plt.show() 



