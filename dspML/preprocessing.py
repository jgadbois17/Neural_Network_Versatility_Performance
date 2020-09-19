#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
import tensorflow as tf 
from keras.utils import to_categorical 
from skimage.transform import downscale_local_mean 
from sktime.forecasting.model_selection import temporal_train_test_split 


''' For Images '''

class image: 
    def flatten_data(X): 
        data = [] 
        for i in range(len(X)): 
            x = X[i].flatten() 
            data.append(x) 
        return np.array(data) 
    
    def classification_data_prep(X, y): 
        n_classes = len(np.unique(y)) 
        y = to_categorical(y, num_classes=n_classes) 
        X = np.expand_dims(X, axis=3)/255. 
        return X, y 
    
    def downsample(X, factors=(2, 2)): 
        for i in range(len(X)): 
            X[i] = downscale_local_mean(X[i], factors) 
        return X 
    
    def reshape(X, size): 
        return tf.image.resize(X, size).numpy()  
    
    def normalize_train(X): 
        m, s = np.mean(X), np.std(X) 
        X = (X - m) / s 
        return X, [m, s] 
    
    def normalize_test(X, norm): 
        return (X - norm[0]) / norm[1] 


''' For Sequences ''' 

def signal_to_np(signal): 
        return np.expand_dims(signal.values.astype(np.float32), axis=1) 

class sequence: 
    def to_numpy(x): 
        return np.array(x).astype(np.float32) 
    
    def reduce_dims(x): 
        return np.squeeze(x) 
    
    def x_sequence(x, time_steps): 
        if type(x) != np.array(x): 
            x = np.array(x) 
        sequences = [] 
        for i in range(len(x)-time_steps+1): 
            sequences.append(x[i:i+time_steps]) 
        sequences = np.array(sequences).astype(np.float32) 
        return sequences 
    
    def xy_sequences(x, time_steps): 
        if type(x) != np.array(x):
            x = signal_to_np(x) 
        sequences = [] 
        next_step = [] 
        for i in range(len(x)-time_steps): 
            sequences.append(x[i:i+time_steps]) 
            next_step.append(x[i+time_steps]) 
        sequences = np.array(sequences).astype(np.float32) 
        next_step = np.array(next_step).astype(np.float32) 
        return sequences, next_step 
    
    def normalize_train(x): 
        m, s = np.mean(x), np.std(x) 
        x = (x-m)/s 
        return x, [m, s] 
    
    def normalize_test(x, norm): 
        return (x-norm[0])/norm[1] 
    
    def to_original_values(x, norm): 
        return x * norm[1] + norm[0] 
    
    def temporal_split(x, fc_hzn): 
        x_train, x_test = temporal_train_test_split(x, test_size=fc_hzn) 
        return x_train, x_test 









