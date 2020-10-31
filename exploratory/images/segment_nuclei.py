#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nuclei Image Data 
Image Segmentation 

Skimage Tutorial: first portion 
My Own Trial: second portion 

NOTE: 
    in your trial, the cells part after threshold might be all you need 
    since it is just a boolean mask 
"""

import numpy as np 
import matplotlib.pyplot as plt 
import scipy.ndimage as ndi 
from skimage import color, feature, filters, segmentation, measure 
from dspML import data 

#%%

# load data 
X, y = data.Nuclei.train() 

#%%

# extract image 
idx = np.random.randint(len(X)) 
img = color.rgb2gray(X[idx]) 

# visualize image 
fig, ax = plt.subplots(figsize=(5, 5))  
ax.imshow(img, cmap='gray') 
ax.set_title('Nuclei Image', size=16) 
plt.show() 

# another visualization 
fig, ax = plt.subplots(figsize=(5, 5)) 
qcs = ax.contour(img, origin='image') 
ax.set_title('Contour Plot of Raw Image', size=16) 
plt.show() 

#%%

''' Estimate the Mitotic Index '''

# threshold 1: separate the nuclei from the background 
# threshold 2: separate the dividing nuclei from non-dividing nuclei 
thresholds = filters.threshold_multiotsu(img, classes=3) 
regions = np.digitize(img, bins=thresholds) 

fig, ax = plt.subplots(ncols=2, figsize=(10, 5)) 
ax[0].imshow(img) 
ax[0].set_title('Original') 
ax[0].axis('off') 
ax[1].imshow(regions) 
ax[1].set_title('Multi-Otsu Thresholding') 
ax[1].axis('off') 
plt.show() 

#%%

# extra since there is overlap (yellow and green) 
cells = img > thresholds[0] 
dividing = img > thresholds[1] 
labeled_cells = measure.label(cells) 
labeled_dividing = measure.label(dividing) 
naive_mi = labeled_dividing.max() / labeled_cells.max() 
print(naive_mi) 
print(labeled_dividing.max()) 
print(labeled_cells.max()) 

fig, ax = plt.subplots(ncols=3, figsize=(12, 5)) 
ax[0].imshow(img) 
ax[0].set_title('original') 
ax[0].axis('off') 
ax[2].imshow(cells) 
ax[2].set_title('All Nuclei?') 
ax[2].axis('off') 
ax[1].imshow(dividing) 
ax[1].set_title('Dividing Nuclei?') 
ax[1].axis('off') 
plt.show() 

#%%

''' Segment Nuclei '''

distance = ndi.distance_transform_edt(cells) 
local_maxi = feature.peak_local_max(distance, indices=False, min_distance=2) 
markers = measure.label(local_maxi) 
segmented_cells = segmentation.watershed(-distance, markers, mask=cells) 

fig, ax = plt.subplots(ncols=2, figsize=(10, 5)) 
ax[0].imshow(cells, cmap='gray') 
ax[0].set_title('Overlapping Nuclei') 
ax[0].axis('off') 
ax[1].imshow(color.label2rgb(segmented_cells, bg_label=0)) 
ax[1].set_title('Segmented Nuclei') 
ax[1].axis('off') 
plt.show() 

#%%

''' Trial: Only One Threshold '''

# extract image 
idx = np.random.randint(len(X)) 
img = color.rgb2gray(X[idx]) 

# visualize image and contour 
fig, ax = plt.subplots(ncols=2, figsize=(10, 5))  
ax[0].imshow(img, cmap='gray') 
ax[0].set_title('Nuclei Image', size=16) 
ax[0].axis('off') 
qcs = ax[1].contour(img, origin='image') 
ax[1].set_title('Contour Plot of Raw Image', size=16) 
ax[1].axis('off') 
plt.show() 

# threshold filter 
threshold = filters.threshold_multiotsu(img, classes=2) 
regions = np.digitize(img, bins=thresholds) 
cells = img > threshold[0] 

# watershed algorithm 
distance = ndi.distance_transform_edt(cells) 
local_maxi = feature.peak_local_max(distance, indices=False, min_distance=2) 
markers = measure.label(local_maxi) 
segmented_cells = segmentation.watershed(-distance, markers, mask=cells) 

# plot results 
fig, ax = plt.subplots(ncols=2, figsize=(10, 5)) 
ax[0].imshow(cells, cmap='gray') 
ax[0].set_title('Threshold Filter Nuclei') 
ax[0].axis('off') 
ax[1].imshow(color.label2rgb(segmented_cells, bg_label=0)) 
ax[1].set_title('Watershed Segmented Nuclei') 
ax[1].axis('off') 
plt.show() 

#%%

def threshold_filter_watershed(img): 
    thresh = filters.threshold_multiotsu(img, classes=2) 
    cells = img > thresh[0] 
    dt = ndi.distance_transform_edt(cells) 
    local_maxi = feature.peak_local_max(dt, indices=False, mindistance=1) 
    markers = measure.label(local_maxi) 
    segmented = segmentation.watershed(-dt, markers, mask=cells) 
    return segmented 






