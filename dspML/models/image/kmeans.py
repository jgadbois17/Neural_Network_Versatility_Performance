#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
from sklearn.cluster import KMeans 
from skimage.transform import resize 

def reshape_preds(preds): 
    x = [] 
    for i in range(len(preds)): 
        img = resize(preds[i], (128, 128, 1), mode='constant', preserve_range=True) 
        x.append(img) 
    return np.array(x) 

def predict_clusters(X, n_clusters=2): 
    shape = X[0].shape 
    preds = [] 
    for i in range(len(X)): 
        img = X[i].reshape(-1, 3) 
        kmeans = KMeans(n_clusters=n_clusters, random_state=7).fit(img) 
        mask = kmeans.cluster_centers_[kmeans.labels_] 
        mask = mask.reshape(shape) 
        preds.append(mask) 
    preds = np.array(preds) 
    return reshape_preds(preds) 

