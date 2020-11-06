#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Classification 
Data: KMNIST Handwritten Characters 
Model: Linear Discriminant Analysis 
"""

from dspML import data 
from dspML.plot import kmnist 
from dspML.preprocessing import image 
from dspML.models.image import lda 
from dspML.evaluation import Classification 

#%%

''' Load Data  '''

# load data 
X_train, y_train, X_test, y_test = data.kmnist() 

# plot images 
plot = kmnist(X_train, y_train, X_test, y_test) 
plot.train_observations() 

# flatten images 
X_train = image.flatten_data(X_train) 
X_test = image.flatten_data(X_test) 

# normalize images 
X_train, norm = image.normalize_train(X_train) 
X_test = image.normalize_test(X_test, norm) 

#%%

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

#%%

# plot of all classes 
plot.all_classes() 

# plot test predictions 
plot.class_predictions(preds=yhat, idx=77) 




