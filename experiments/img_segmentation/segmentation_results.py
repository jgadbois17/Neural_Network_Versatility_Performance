#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Segmentation 
Data: Nuclei Data | Kaggle Data Science Bowl 2018 
Model 1: K-Means Clustering 
Model 2: U-Net Fully Convolutional Network 
"""

import numpy as np 
import matplotlib.pyplot as plt 

from dspML import data 
from dspML.preprocessing import split_data 
from dspML.models.image import unet, kmeans as km 

#%%

''' Nuclei Data '''

X_train, y_train = data.Nuclei.train() 
X_train, X_test, y_train, y_test = split_data(X_train, y_train, test_size=0.1) 

#%%

''' K-Means Clustering Predictions '''

# predict test masks 
masks_kmeans = km.predict_clusters(X_test) 

#%%

''' U-Net Convolutional Network '''

# load model and make predictions 
model = unet.load_uNet() 
masks_unet = unet.predict_segmentation_masks(model, X_test) 



