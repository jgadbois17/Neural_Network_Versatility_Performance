#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Filtering with PIL 
"""

import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
from PIL import Image, ImageFilter as Filt 
from dspML import data 

def plot_img(img, title=''): 
    plt.figure(figsize=(6,6)) 
    plt.imshow(img, cmap='gray') 
    plt.title(title, size=14) 
    plt.show() 

def filter_results(images, filt, title=''): 
    plt.figure(figsize=(10, 10)) 
    for i in range(9): 
        plt.subplot(3,3,(i+1)) 
        plt.imshow(images[i].filter(filt), cmap='gray') 
        plt.axis('off') 
    plt.suptitle(title, size=16) 
    plt.tight_layout() 

#%%

''' Load KMNIST Dataset '''

# load full training data 
X, y, _,_= data.kmnist() 

# extract images as PIL images 
images = X[:9]
images = [Image.fromarray(images[i]) for i in range(len(images))] 

# create single PIL image 
im = images[0] 

#%%

''' Filtering Single Images '''

# plot original image 
plot_img(im, title='Original Image') 

# size based filters 
plot_img(im.filter(Filt.MinFilter(size=3)), title='Minimum Filter') 
plot_img(im.filter(Filt.MaxFilter(size=3)), title='Maximum Filter') 
plot_img(im.filter(Filt.MedianFilter(size=3)), title='Median Filter') 
plot_img(im.filter(Filt.ModeFilter(size=3)), title='Mode Filter') 

# radius based filters 
plot_img(im.filter(Filt.GaussianBlur(radius=2)), title='Gaussian Blur Filter') 
plot_img(im.filter(Filt.BoxBlur(radius=2)), title='Box Blur Filter') 

# feature detecting filters 
plot_img(im.filter(Filt.SHARPEN()), title='Sharpen Filter')
plot_img(im.filter(Filt.EDGE_ENHANCE_MORE()), title='Edge Enhance More Filter') 
plot_img(im.filter(Filt.FIND_EDGES()), title='Edge Finder Filter') 
plot_img(im.filter(Filt.SMOOTH_MORE()), title='Smoothing More Filter') 

#%%

''' Filtering Multiple Images '''

# size based filters 
size = 3 
filter_results(images, filt=Filt.MinFilter(size), title='Minimum Filter') 
filter_results(images, filt=Filt.MaxFilter(size), title='Maximum Filter') 
filter_results(images, filt=Filt.MedianFilter(size), title='Median Filter') 
filter_results(images, filt=Filt.ModeFilter(size), title='Mode Filter') 

#%%

# radius based filters 
radius = 1.75
filter_results(images, filt=Filt.GaussianBlur(radius), title='Gaussian Blur Filter') 
filter_results(images, filt=Filt.BoxBlur(radius), title='Box Blur Filter') 

#%%

# feature detecting filters 
filter_results(images, filt=Filt.EDGE_ENHANCE_MORE(), title='Edge Enhancement Filter') 
filter_results(images, filt=Filt.FIND_EDGES(), title='Edge Finder Filter') 
filter_results(images, filt=Filt.SHARPEN(), title='Sharpening Filter') 
filter_results(images, filt=Filt.SMOOTH(), title='Smoothing Filter') 

#%%

''' Applying Multiple Filters '''

def feature_detection(im, filt, title=''): 
    plt.figure(figsize=(10, 5)) 
    plt.subplot(1,3,1) 
    plt.imshow(im.filter(Filt.FIND_EDGES()).filter(filt), cmap='gray') 
    plt.axis('off'), plt.title('Edge Finder') 
    plt.subplot(1,3,2) 
    plt.imshow(im.filter(Filt.SHARPEN()).filter(filt), cmap='gray') 
    plt.axis('off'), plt.title('Sharpen') 
    plt.subplot(1,3,3) 
    plt.imshow(im.filter(Filt.SMOOTH()).filter(filt), cmap='gray') 
    plt.axis('off'), plt.title('Smooth') 
    plt.suptitle(title, size=16) 
    plt.tight_layout() 

feature_detection(im, filt=Filt.MinFilter(3), title='Minimum Filtering') 
feature_detection(im, filt=Filt.MaxFilter(3), title='Maximum Filtering') 
feature_detection(im, filt=Filt.MedianFilter(3), title='Median Filtering') 

#%%

''' Filter Trials '''

plot_img(im, title='Original Image') 
plot_img(im.filter(Filt.FIND_EDGES()), title='Edge Finder Filter') 
plot_img(im.filter(Filt.MinFilter(size=3)), title='Minimum Filter') 
plot_img(im.filter(Filt.MaxFilter(size=3)), title='Maximum Filter') 
feature_detection(im, filt=Filt.MinFilter(3), title='Minimum Filtering') 
feature_detection(im, filt=Filt.MaxFilter(3), title='Maximum Filtering') 



















