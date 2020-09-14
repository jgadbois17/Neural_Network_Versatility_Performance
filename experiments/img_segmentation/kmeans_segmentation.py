#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Segmentation 

Data: Nuclei Data | Kaggle Data Science Bowl 2018 
Model: K-Means Clustering 
"""

from dspML import data 
from dspML.plot import Nuclei 
from dspML.models.image import kmeans 
from dspML import evaluation as ev 

#%%

''' Load Data '''

# load data 
X_train, y_train = data.Nuclei.train() 
X_test = data.Nuclei.test() 

# plot observations 
plot = Nuclei(X_train, y_train, X_test) 
plot.observations() 

#%%

''' K-Means Clustering Segmentation '''

# training predictions 
p_mask_train = kmeans.KMeans_segmentation(X_train) 
p_mask_train = kmeans.reshape_preds(p_mask_train) 

# plot training predictions 
plot.train_predictions(p_mask_train) 

# evaluation 
km_eval = ev.Segmentation(y_train, p_mask_train) 
km_eval.DICE() 
km_eval.soft_DICE() 

# test predictions 
p_mask = kmeans.KMeans_segmentation(X_test) 
p_mask = kmeans.reshape_preds(p_mask) 

# plot test predictions 
plot.test_pred_masks(p_mask) 









