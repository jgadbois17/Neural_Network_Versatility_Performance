#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
import matplotlib.pyplot as plt 
from skimage import color, feature, filters, segmentation, measure 
from dspML import data 

#%%

''' Nuclei Data '''

X, y = data.Nuclei.train() 

#%%

def plot_img(x, cmap='gray'): 
    plt.imshow(x, cmap=cmap) 
    plt.axis('off') 

#%%

# extract image 
idx = np.random.randint(len(X)) 
img = color.rgb2gray(X[idx]) 

plt.hist(img) 
plt.show() 




