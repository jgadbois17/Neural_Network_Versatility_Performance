#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
import matplotlib.pyplot as plt 

from dspML import data 
from dspML.plot import kmnist 
from dspML.preprocessing import image 
from dspML.models.image import cnn, lda 
from dspML.evaluation import Classification 

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



