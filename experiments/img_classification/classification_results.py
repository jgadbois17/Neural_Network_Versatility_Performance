#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Classification 
Data: KMNIST Handwritten Characters 
Model 1: Linear Discriminant Analysis 
Model 2: Convolutional Neural Network 
"""

import numpy as np 
import matplotlib.pyplot as plt 

from dspML import data 
from dspML.preprocessing import image 
from dspML.models.image import cnn, lda 

#%%

''' Linear Discriminant Analysis '''

# load data 
X_train, y_train, X_test, y_test = data.kmnist() 

# preprocessing data 
X_train = image.flatten_data(X_train) 
X_train, norm = image.normalize_train(X_train) 
X_test = image.flatten_data(X_test) 
X_test = image.normalize_test(X_test, norm) 

# define model and get predictions 
model_lda = lda.LDA(X_train, y_train) 
preds_lda = model_lda.predict(X_test) 

#%%

''' Convolutional Neural Network '''

# load data 
X_train, y_train, X_test, y_test = data.kmnist() 

# preprocessing data 
n_classes = len(np.unique(y_test)) 
X_train, y_train = image.classification_data_prep(X_train, y_train) 
X_test,_ = image.classification_data_prep(X_test, y_test) 

# define and fit model and get predictions 
model_cnn = cnn.ConvNet(in_shape=X_train.shape[1:], n_classes=n_classes) 
model_cnn.fit(X_train, y_train, epochs=10, validation_split=0.1, shuffle=True) 
preds_cnn = cnn.predictions(model_cnn, X_test) 

#%%

''' Plot Predictions Both Models '''

def plot_predictions(X, y, p_lda, p_cnn): 
    plt.figure(figsize=(12, 5)) 
    for i in range(3): 
        idx = np.random.randint(len(X)) 
        plt.subplot(1, 3, (i+1)) 
        plt.imshow(X[idx], cmap='gray') 
        plt.axis('off') 
        plt.title('True: {} | LDA: {} | CNN: {}'
                  .format(y[idx], p_lda[idx], p_cnn[idx]), size=16) 
    plt.suptitle('Class Predictions', size=20) 
    plt.tight_layout() 

plot_predictions(X_test, y_test, preds_lda, preds_cnn) 

