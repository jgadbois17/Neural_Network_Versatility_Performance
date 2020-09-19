#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class to apply filters to an entire image dataset 
"""

from dspML import data, plot 
from dspML.preprocessing import image 
from dspML.models.image import lda 
from dspML.evaluation import Classification 

import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
from PIL import Image, ImageFilter as Filt 


''' Load KMNIST Image Dataset '''

# load data 
X_train, y_train, X_test, y_test = data.kmnist() 


''' Filtering '''

class Filter(): 
    def __init__(self, images): 
        self.n = len(images) 
        self.images = [Image.fromarray(images[i]) for i in range(len(images))] 
    
    def single_filter(self, filt): 
        X = [] 
        for i in range(self.n): 
            img = self.images[i].filter(filt) 
            X.append(np.asarray(img)) 
        return np.array(X) 
    
    def double_filter(self, filt1, filt2): 
        X = [] 
        for i in range(self.n): 
            img = self.images[i].filter(filt1).filter(filt2) 
            X.append(np.asarray(img)) 
        return np.array(X) 

# filter transformations 
train_filt = Filter(images=X_train) 
test_filt = Filter(images=X_test) 

# apply multiple features 
filts = [Filt.EMBOSS(), Filt.MaxFilter(3)] 
X_train = train_filt.double_filter(filts[0], filts[1]) 
X_test = test_filt.double_filter(filts[0], filts[1]) 

# plot filtered images 
km_plot = plot.kmnist(X_train, y_train, X_test, y_test) 
km_plot.train_observations(cmap='gray')  
km_plot.class_means() 

