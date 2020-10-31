#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Segmentation 
Data: Nuclei Data | Kaggle Data Science Bowl 2018 
Model: K-Means Clustering 
"""

from dspML import data 
from dspML.plot import Nuclei, plot_observed_segmentations 
from dspML.preprocessing import split_data 
from dspML.models.image import kmeans 
from dspML.evaluation import SegmentationMetrics 

#%%

''' Nuclei Data '''

# load data 
X, y = data.Nuclei.train() 
test = data.Nuclei.test() 

# plot observations 
plot_observed_segmentations(X, y) 

# split data 
X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.1) 
plot = Nuclei(X_train, y_train, X_test, y_test, test) 

# data dimensions 
print('Training Data Dimensions:') 
print('X_train = ', X_train.shape, '\ny_train = ', y_train.shape) 
print('\nTesting Data Dimensions:') 
print('X_test = ', X_test.shape, '\ny_test = ', y_test.shape) 

#%%

''' K-Means Clustering Segmentation '''

# predictions 
p_train = kmeans.predict_clusters(X_train) 
p_test = kmeans.predict_clusters(X_test) 

# plot predictions 
plot.predicted_mask(p_train) 
plot.predicted_mask(p_test) 

# evaluate model 
metrics = SegmentationMetrics(X_train, y_train, X_test, y_test) 
metrics.DiceCoefficient(p_train) 
metrics.softDice_loss(p_train) 
print(' ') 
metrics.DiceCoefficient(p_test) 
metrics.softDice_loss(p_test) 

#%%

''' Predicting Unknown Masks '''

# predict segmentation masks 
masks = kmeans.predict_clusters(test) 
plot.new_random_segmentations(masks) 


