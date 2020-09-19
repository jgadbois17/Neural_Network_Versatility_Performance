#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dspML import data, plot 
from dspML.preprocessing import image 
from dspML.models.image import lda 
from dspML.evaluation import Classification 

import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 

def plot_images(X, y, idx=[0, 1, 2], cmap='gray'): 
    plt.figure(figsize=(10, 6)) 
    plt.subplot(1,3,1) 
    plt.imshow(X[idx[0]], cmap=cmap) 
    plt.title('Label: {}'.format(idx[0])) 
    plt.subplot(1,3,2) 
    plt.imshow(X[idx[1]], cmap=cmap) 
    plt.title('Label: {}'.format(idx[1])) 
    plt.subplot(1,3,3) 
    plt.imshow(X[idx[2]], cmap=cmap) 
    plt.title('Label: {}'.format(idx[2])) 
    plt.tight_layout() 

def fft2d_filter(img, keep_frac=0.1): 
    fft = tf.signal.fft2d(img) 
    fft2 = fft.numpy().copy() 
    r,c = fft2.shape[:2] 
    fft2[int(r*keep_frac):int(r*(1-keep_frac))] = 0 
    fft2[:, int(c*keep_frac):int(c*(1-keep_frac))] = 0 
    return tf.signal.ifft2d(fft2).numpy().real 

def fft_filter_data(data, keep_frac=0.1): 
    x_filt = [] 
    for i in range(len(data)): 
        x = fft2d_filter(data[i], keep_frac) 
        x_filt.append(x) 
    return np.array(x_filt) 

#%%

''' Load Data '''

# load data 
X_train, y_train, X_test, y_test = data.kmnist() 

# initial images 
plot_images(X_train, y_train) 

''' Filter Data '''

# filter images 
X_train = fft_filter_data(X_train, keep_frac=0.75) 
X_test = fft_filter_data(X_test, keep_frac=0.75) 

# filtered/blurred images 
plot_images(X_train, y_train) 

# plot data 
kmnist_plot = plot.kmnist(X_train, y_train, X_test, y_test) 
kmnist_plot.train_observations() 
kmnist_plot.class_means() 

# flatten and normalize data 
X_train = image.flatten_data(X_train) / 255. 
X_test = image.flatten_data(X_test) / 255. 


''' Linear Discriminant Analysis '''

# linear discriminant analysis 
model = lda.LDA(X_train, y_train) 

# predictions 
y_pred = model.predict(X_train) 

# evaluation on training data 
ev_train = Classification(y_train, y_pred) 
ev_train.accuracy() 
ev_train.confusion_matrix() 
ev_train.classification_report() 

# test predictions 
yhat = model.predict(X_test) 

# evaluation 
ev_test = Classification(y_test, yhat) 
ev_test.accuracy() 
ev_test.confusion_matrix() 
ev_test.classification_report() 





















