#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Segmentation 
Data: Nuclei Data | Kaggle Data Science Bowl 2018 
Model: U-Net Fully Convolutional Network 
"""

from dspML import data 
from dspML.plot import Nuclei, plot_observed_segmentations 
from dspML.preprocessing import split_data 
from dspML.models.image import unet 
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

''' U-Net Image Segmentationn '''

# load model 
model = unet.load_uNet() 
model.summary() 

# predictions 
p_train = unet.predict_segmentation_masks(model, X_train) 
p_test = unet.predict_segmentation_masks(model, X_test) 

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
masks = unet.predict_segmentation_masks(model, test) 
plot.new_random_segmentations(masks) 


