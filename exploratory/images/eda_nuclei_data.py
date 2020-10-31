#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nuclei Image Data 
Explore Data and Segmentation Models 
"""

import numpy as np 
import matplotlib.pyplot as plt 
from skimage import color, filters, feature, measure, segmentation 
from skimage.morphology import reconstruction 
from scipy.ndimage import distance_transform_edt, gaussian_filter 
from sklearn.cluster import KMeans 
from dspML import data 

#%%

''' Nuclei Data ''' 

X, y = data.Nuclei.train() 

#%%

''' Useful Functions '''

def desciptives(x): 
    print('Pixel Value Statistics:') 
    print('Mean : {}'.format(np.around(np.mean(x), 4))) 
    print('Min  : {}'.format(np.around(np.min(x), 4))) 
    print('Max  : {}'.format(np.around(np.max(x), 4))) 


''' Plotting Functions '''

def plot_img(x, title='Image', cmap='gray'): 
    plt.imshow(x, cmap=cmap) 
    plt.title(title, size=14) 

def display_one(x, title='Image', figsize=(5, 5), cmap='gray'): 
    plt.figure(figsize=figsize) 
    plot_img(x, title, cmap) 
    plt.show() 


#%%

''' Extract Image '''

idx = np.random.randint(len(X)) 
img = color.rgb2gray(X[idx]) 
display_one(img, title='Original Image') 
desciptives(img) 

#%%

''' Random Walker Segmentation ''' 

# range of pixel values: (0.0386, 0.4627) 
markers = np.zeros(img.shape, dtype=np.uint) 
markers[img < 0.03] = 1 
markers[img > 0.43] = 2 

# random walker algorithm 
labels = segmentation.random_walker(img, markers, beta=10, mode='bf') 

# plot results 
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 3.2), sharex=True, sharey=True) 
ax1.imshow(img, cmap='gray') 
ax1.axis('off') 
ax1.set_title('Original Image') 
ax2.imshow(markers, cmap='magma') 
ax2.axis('off') 
ax2.set_title('Markers') 
ax3.imshow(labels, cmap='gray') 
ax3.axis('off') 
ax3.set_title('Segmentation') 
fig.tight_layout() 
plt.show() 

#%%

''' Chan-Vese Segmentation Algorithm (Level Set) ''' 

# CV Algorithm (play with parameters) 
cv = segmentation.chan_vese(
    img, mu=0.5, lambda1=1, lambda2=1, tol=1e-3, max_iter=200, dt=0.5, 
    init_level_set='checkerboard', extended_output=True) 

# plot 
fig, axes = plt.subplots(2, 2, figsize=(8, 8)) 
ax = axes.flatten() 
ax[0].imshow(img, cmap='gray') 
ax[0].set_axis_off() 
ax[0].set_title('Original Image') 
ax[1].imshow(cv[0], cmap='gray') 
ax[1].set_axis_off() 
ax[1].set_title('Chan-Vese Seg - {} iterations'.format(len(cv[2]))) 
ax[2].imshow(cv[1], cmap='gray') 
ax[2].set_axis_off() 
ax[2].set_title('Final Level Set')  
ax[3].plot(cv[2]) 
ax[3].set_title('Evolution of Energy over Iterations') 
fig.tight_layout() 
plt.show() 
  
#%%

''' Filtering Regional Maxima ''' 

# apply filter (note: image must be float) 
img1 = gaussian_filter(img, 1) 

seed = np.copy(img1) 
seed[1:-1, 1:-1] = img1.min() 
mask = img1 

dilated = reconstruction(seed, mask, method='dilation') 

# plot 
fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True) 
ax0.imshow(img1, cmap='gray') 
ax0.set_title('Original') 
ax0.axis('off') 
ax1.imshow(dilated, vmin=img1.min(), vmax=img1.max(), cmap='gray') 
ax1.set_title('Dilated') 
ax1.axis('off') 
ax2.imshow(img1-dilated, cmap='gray') 
ax2.set_title('Image-Dilated') 
ax2.axis('off') 
fig.tight_layout() 






















