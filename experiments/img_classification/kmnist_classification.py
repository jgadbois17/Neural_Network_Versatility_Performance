#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Classification 

Data: 
    KMNIST Dataset 

Models: 
    Convolutional Neural Network 
    Linear Discriminant Analysis 
"""

import numpy as np 
from dspML.data import data 
from dspML.image import utils_img as utils 
from dspML.image import plot_img as plot 
from dspML.image.models import fit_img as fit 
from dspML.image.models.classification import LDA, CNN_classifier 

#%%

''' Load KMNIST Data '''

# load data 
X_train, y_train, X_test, y_test = data.load_kmnist() 

# plot images 
plot.kmnist_images(X_train, y_train) 
plot.kmnist_images(X_train, y_train) 
plot.kmnist_images(X_train, y_train) 

# plot class averages 
plot.kmnist_class_means(X_train, y_train, figsize=(10, 5))  

#%%

''' Linear Discriminant Analysis ''' 

# load data 
X_train, y_train, X_test, y_test = data.load_kmnist() 

# preprocess data 
X_train = utils.flatten_data(X_train) 
X_test = utils.flatten_data(X_test) 

# LDA 
lda = LDA(X_train, y_train, solver='lsqr') 
lda_preds = lda.predict(X_test) 
utils.classification_summary(y_test, lda_preds) 

#%%

''' Convolutional Network '''

# load data 
X_train, y_train, X_test, y_test = data.load_kmnist() 

# preprocess data 
n_classes = len(np.unique(y_test)) 
X_train, y_train = utils.preprocess_classification_data(X_train, y_train) 
X_test,_ = utils.preprocess_classification_data(X_test, y_test) 

# CNN 
cnn = CNN_classifier(in_shape=X_train.shape[1:], n_classes=n_classes) 
cnn.summary() 
cnn_hist = fit.classifier(cnn, X_train, y_train) 
cnn_preds = utils.cnn_preds(cnn, X_test) 
utils.classification_summary(y_test, cnn_preds) 

#%%

''' Compare Results '''

utils.compare_results(y_test, lda_preds, cnn_preds) 

