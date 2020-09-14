#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Segmentation 

Data: 
    Nuclei Data from Kaggle's Data Science Bowl 2018 

Models: 
    U-Net Convolutional Network 
    K-Means Clustering Segmentation 
"""

from dspML.data import data 
from dspML.image import utils_img as utils 
from dspML.image import plot_img as plot 
from dspML.image.models.segmentation import load_uNet, KMeans_segmentation 

#%%

''' Load Nuclei Data '''

# load data 
X_train, y_train = data.load_nuclei_train() 
X_test = data.load_nuclei_test() 

# plot observations 
plot.nuclei_observations(X_train, y_train) 
plot.nuclei_observations(X_train, y_train) 
plot.nuclei_observations(X_train, y_train) 

#%%

''' K-Means Cluster Segmentation '''

# predicted training masks 
X_tr_km = utils.greyscale(X_train) 
km_preds_train = KMeans_segmentation(X_tr_km)  
km_preds_train = utils.resize_preds_km(km_preds_train) 
plot.nuclei_train_preds(X_train, y_train, km_preds_train) 

# predicted testing masks 
X_te_km = utils.greyscale(X_test) 
km_preds_test = KMeans_segmentation(X_te_km)  
km_preds_test = utils.resize_preds_km(km_preds_test) 
plot.nuclei_test_preds(X_test, km_preds_test) 

#%%

''' U-Net Segmentation '''

# load model 
path = 'dspML/image/models/fitted_models/unet_fitted_model_2.h5' 
unet = load_uNet(path) 
unet.summary() 

# training predictions 
unet_preds_train = utils.unet_preds(unet, X_train) 
plot.nuclei_train_preds(X_train, y_train, unet_preds_train) 

# testing predictions 
unet_preds_test = utils.unet_preds(unet, X_test) 
plot.nuclei_test_preds(X_test, unet_preds_test) 

#%%

''' DICE Coefficient and Soft DICE Losses '''

# DICE coefficients 
utils.DICE_Coefficients(y_train, km_preds_train, unet_preds_train) 

# Soft DICE losses 
utils.Soft_DICE_Loss(y_train, km_preds_train, unet_preds_train) 

#%%

import numpy as np 
import matplotlib.pyplot as plt 

def plot_test_preds(x, preds_u, preds_k, idx=[0, 1, 2]): 
    plt.figure(figsize=(10, 10)) 
    for i in range(3): 
        plt.subplot(3, 3, (i+1)) 
        plt.imshow(x[idx[i]]) 
        plt.axis('off') 
        plt.title('Image {}'.format(idx[i]), size=14) 
        plt.subplot(3, 3, (i+4)) 
        plt.imshow(np.squeeze(preds_u[idx[i]]), cmap='gray') 
        plt.axis('off') 
        plt.title('U-Net Predicted Mask', size=14) 
        plt.subplot(3, 3, (i+7)) 
        plt.imshow(np.squeeze(preds_k[idx[i]]), cmap='gray')  
        plt.axis('off') 
        plt.title('K-Means Predicted Mask', size=14) 
    plt.tight_layout() 

idx = [11, 26, 42]
plot_test_preds(X_test, unet_preds_test, km_preds_test, idx) 
#plt.savefig('fig4.3_pred_segs.png')








