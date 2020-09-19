#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
import pandas as pd 
from keras import Sequential, layers, optimizers 
from keras.callbacks import EarlyStopping 

def Recurrent(input_shape=(None, 1), name='Recurrent_Forecaster'): 
    model = Sequential(name=name) 
    model.add(layers.Input(shape=input_shape)) 
    
    model.add(layers.GRU(units=20, return_sequences=True, name='GRU')) 
    
    model.add(layers.TimeDistributed(layers.Dense(1), name='FC')) 
    model.compile(optimizer=optimizers.Adam(), loss='mse') 
    return model 

def Convolutional(time_steps, input_shape=(None, 1), name='Convolutional_Forecaster'): 
    model = Sequential(name=name) 
    model.add(layers.Input(shape=input_shape)) 
    
    model.add(layers.Conv1D(32, kernel_size=3, activation='relu', name='Conv1D')) 
    model.add(layers.MaxPool1D(2, padding='same', name='MaxPool')) 
    model.add(layers.Dense(time_steps, activation='sigmoid', name='Dense')) 
    
    model.add(layers.TimeDistributed(layers.Dense(1), name='FC')) 
    model.compile(optimizer=optimizers.Adam(), loss='mse') 
    return model 

def fit(model, x, y, batch_size=32, epochs=1000, verbose=1, shuffle=True, patience=50): 
    cb = [EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)] 
    hist = model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=verbose, 
                     callbacks=cb, validation_split=0.1, shuffle=shuffle) 
    return hist 

def predict_forecast(model, train, steps): 
    forecast = [] 
    n = train.shape[1]
    train = np.squeeze(train[-1]) 
    for i in range(steps): 
        x = np.expand_dims(train[-n:], axis=(0, 2)) 
        pred = model.predict(x) 
        forecast.append(pred[0][0]) 
        train = np.append(train, pred[0][0]) 
    return pd.Series(forecast) 



