#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Classification 

Data: KMNIST Handwritten Characters 
Model: Convolutional Neural Network 
"""

import numpy as np 
from dspML import data 
from dspML.plot import kmnist 
from dspML.preprocessing import image 
from dspML.models.image import cnn 
from dspML import evaluation as ev 


''' Load Data '''

# load data 
X_train, y_train, X_test, y_test = data.kmnist() 

# plot data 
plot = kmnist(X_train, y_train, X_test, y_test) 
plot.train_observations() 

# preprocessing data 
n_classes = len(np.unique(y_test)) 
X_train, y_train = image.classification_data_prep(X_train, y_train) 
X_test,_ = image.classification_data_prep(X_test, y_test) 


''' Convolutional Network '''

# define model 
model = cnn.ConvNet(in_shape=X_train.shape[1:], n_classes=n_classes) 
model.summary() 

# fit model 
model.fit(X_train, y_train, epochs=10, validation_split=0.1, shuffle=True) 

# train accuracy 
_,train_accuracy = model.evaluate(X_train, y_train) 

# test predictions 
yhat = cnn.predictions(model, X_test) 

# evaluate model 
ev_test = ev.Classification(y_test, yhat) 
ev_test.accuracy() 
ev_test.confusion_matrix() 
ev_test.classification_report() 











