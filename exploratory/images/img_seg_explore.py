#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nuclei Image Data 
Image Segmentation Trials and Exploration 
"""

import numpy as np 
import matplotlib.pyplot as plt 
from skimage import color, filters, feature, measure, segmentation
from scipy.ndimage import distance_transform_edt 
from sklearn.cluster import KMeans 
from dspML import data 

# load data 
X, y = data.Nuclei.train() 

#%%

''' 
Image idx to use: 66, 405, 521, 198 
Images must be in gray scale 
'''

# extract a signal image and convert to gray scale 
idx = 405 
img = color.rgb2gray(X[idx]) 

plt.imshow(img, cmap='gray') 
plt.title('Original Image', size=16) 
plt.show() 


''' 
Smooth (Denoise) Image 
selem >> filter dimensions 
'''

# apply filter (selem: filter size >> same dimensions as image) 
img_denoised = filters.median(img, selem=np.ones((5, 5))) 

plt.figure(figsize=(10, 5)) 
plt.subplot(121) 
plt.imshow(img) 
plt.title('Original Image', size=16) 
plt.subplot(122) 
plt.imshow(img_denoised) 
plt.title('Denoised Image', size=16) 
plt.show() 

''' 
Edge Detection 
sigma >> lower number = more sensitive to detecting edges 
'''

# detect edges 
edges = feature.canny(img, sigma=1) 

plt.imshow(edges) 
plt.title('Detected Edges', size=16) 
plt.show() 


''' 
Watershed Algorithm 
Convert the edges image into a landscape 
Darker regions of the inverse mean we are closer to an edge 
'''

# compute euclidena distance transform 
dt = distance_transform_edt(edges) 
#plt.imshow(dt) 

# compute inverse euclidena distance transform 
# inverse >> flip the foreground and background 
dt = distance_transform_edt(~edges) 
plt.imshow(dt) 
plt.title('Euclidean Distance Transform', size=16) 
plt.show() 


'''
Fill the watershed landscape computed above 
Find the locations of the "fountains" by finding local maxima 
'''

local_max = feature.peak_local_max(dt, indices=False, min_distance=5) 
plt.imshow(local_max, cmap='gray') 
plt.title('Watershed', size=16) 
plt.show() 


# to get the positions of the peaks, repeat the above with indices=True 
peak_idx = feature.peak_local_max(dt, indices=True, min_distance=5) 
print('First 5: \n\n {}'.format(peak_idx[:5])) 

# plot all peaks 
plt.plot(peak_idx[:, 1], peak_idx[:, 0], 'r.') 
plt.imshow(dt) 
plt.title('Detected Peaks', size=16) 
plt.show() 


''' Label Features '''

# get markers (labels for each peak) 
markers = measure.label(local_max) 

# watershed function 
labels = segmentation.watershed(-dt, markers) 
plt.imshow(segmentation.mark_boundaries(img, labels)) 
plt.title('Labeled Peaks', size=16) 
plt.show() 

# different visualization 
plt.imshow(color.label2rgb(labels, image=img)) 
plt.title('Colored Labeled Peaks', size=16) 
plt.show() 

# another visualization of the segmentation 
# kind='avg' >> instead of colors, use average pixel values in the segmentation areas 
plt.imshow(color.label2rgb(labels, image=img, kind='avg'), cmap='gray') 
plt.title('Average Labeled Peaks', size=16) 
plt.show() 


'''
Regions: NEED TO LOOK UP 
'''

# compute regions 
regions = measure.regionprops(labels, intensity_image=img) 

# compute mean intensity for each of the regions 
region_means = [r.mean_intensity for r in regions] 
plt.hist(region_means, bins=20) 
plt.title('Pixel Value Histogram', size=16) 
plt.show() 


''' Cluster the Regions '''

# reshape region means to work with KMeans 
region_means = np.array(region_means).reshape(-1, 1) 

# define and fit model 
model = KMeans(n_clusters=2) 
model.fit(region_means) 
print(model.cluster_centers_) 


# predict labels for each of the regions 
bg_fg_labels = model.predict(region_means) 
print(bg_fg_labels) 

# label image appropriately 
classified_labels = labels.copy() 
for bg_fg, region in zip(bg_fg_labels, regions): 
    classified_labels[tuple(region.coords.T)] = bg_fg 

# plot results 
plt.figure(figsize=(10, 5)) 
plt.subplot(121) 
plt.imshow(img) 
plt.title('Original Image', size=16) 
plt.subplot(122) 
plt.imshow(color.label2rgb(classified_labels, image=img)) 
plt.title('Segmentation', size=16) 
plt.show() 








