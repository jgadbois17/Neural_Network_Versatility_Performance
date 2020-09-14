#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Segmentation 

Data: Nuclei Data | Kaggle Data Science Bowl 2018 
Model: K-Means Clustering 
"""

from dspML import data 
from dspML.plot import Nuclei 
from dspML.models.image import unet 
from dspML import evaluation as ev 

#%%

''' Load Data '''

# load data 
X_train, y_train = data.load.Nuclei.train() 
X_test = data.load.Nuclei.test() 

# plot observations 
plot = Nuclei(X_train, y_train, X_test) 
plot.observations() 

#%%

''' U-Net Image Segmentationn '''

# load model 
path = 'dspML/models/image/fitted/unet_fitted_model_2.h5' 
model = unet.load_uNet(path) 
model.summary() 

# training predictions 
p_mask_train = unet.pred_masks(model, X_train) 

# plot training predictions 
plot.train_predictions(p_mask_train) 

# evaluate model 
un_eval = ev.Segmentation(y_train, p_mask_train) 
un_eval.DICE() 
un_eval.soft_DICE() 

# testing predictions 
p_mask = unet.pred_masks(model, X_test) 

# plot test predictions 
plot.test_pred_masks(p_mask) 


