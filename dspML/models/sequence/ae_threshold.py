#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
from tensorflow import nn 
from keras import Sequential, layers, optimizers 

def Encoder(input_shape): 
    model = Sequential(name='Encoder') 
    model.add(layers.Input(shape=input_shape)) 
    model.add(layers.Conv1D(64, 7, strides=2, padding='same', activation=nn.relu)) 
    model.add(layers.Dropout(0.4)) 
    model.add(layers.Conv1D(32, 7, strides=2, padding='same', activation=nn.relu)) 
    model.add(layers.Dropout(0.3)) 
    model.add(layers.Conv1D(1, 7, strides=1, padding='same', activation=nn.relu)) 
    model.add(layers.MaxPool1D(pool_size=2, padding='same')) 
    return model 

def Decoder(input_shape): 
    model = Sequential(name='Decoder') 
    model.add(layers.Input(shape=input_shape)) 
    model.add(layers.Conv1DTranspose(1, 7, strides=2, padding='same', activation=nn.relu)) 
    model.add(layers.Conv1DTranspose(32, 7, strides=2, padding='same', activation=nn.relu)) 
    model.add(layers.Conv1DTranspose(64, 7, strides=2, padding='same', activation=nn.relu)) 
    model.add(layers.Conv1DTranspose(1, 7, strides=1, padding='same')) 
    return model 

def ConvAE(input_shape): 
    encoder = Encoder(input_shape) 
    d_shape = ((input_shape[0]//(8)), input_shape[1]) 
    decoder = Decoder(d_shape) 
    model = Sequential([encoder, decoder], name='Convolutional_Autoencoder') 
    model.compile(optimizer=optimizers.Adam(), loss='mae') 
    return model 


def anomaly_threshold(model, x): 
    preds = model.predict(x) 
    mae = np.mean(np.abs(preds - x), axis=1)  
    threshold = np.max(mae) 
    print('Max MAE threshold = {}'.format(threshold)) 
    return threshold 

def predict_anomalies(model, x, threshold): 
    preds = model.predict(x) 
    mae = np.mean(np.abs(preds - x), axis=1).reshape((-1)) 
    anomalies = (mae > threshold).tolist() 
    return anomalies 

def anomaly_ids(anomalies, x, time_steps): 
    anom_ids = [] 
    for idx in range(time_steps-1, len(x)-time_steps+1): 
        ts = range(idx-time_steps+1, idx) 
        if all([anomalies[j] for j in ts]): 
            anom_ids.append(idx) 
    return anom_ids 


