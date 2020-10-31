#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
import pandas as pd 
from keras import Sequential, layers, optimizers 
from keras.callbacks import EarlyStopping, ModelCheckpoint 
from keras.models import load_model 

def GRUNet(input_shape=(None, 1), name='GRU_Recurrent_Network'):
    model = Sequential(name=name) 
    model.add(layers.Input(shape=input_shape)) 
    model.add(layers.GRU(32, return_sequences=True)) 
    model.add(layers.GRU(64, return_sequences=False)) 
    model.add(layers.Dense(1)) 
    model.compile(loss='mse', optimizer=optimizers.Adam()) 
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

def fit(model, x, y, path=None, batch_size=32, epochs=1000, verbose=1, shuffle=True, 
        patience=25, val_split=0.1):
    es = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True) 
    if path is not None: 
        base_path = 'dspML/models/sequence/fitted/' 
        mcp = ModelCheckpoint(base_path+path, monitor='val_loss', save_best_only=True) 
        cb = [es, mcp] 
    else: 
        cb = [es] 
    hist = model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=verbose, 
                     callbacks=cb, validation_split=val_split, shuffle=shuffle) 
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

def load_humidity(recurrent=True): 
    if recurrent: 
        path = 'dspML/models/sequence/fitted/GRU_humidity.h5' 
    else: 
        path = 'dspML/models/sequence/fitted/cnn_humidity.h5' 
    return load_model(path) 

def load_wind_speed(recurrent=True): 
    if recurrent: 
        path = 'dspML/models/sequence/fitted/GRU_wind_speed.h5' 
    else: 
        path = 'dspML/models/sequence/fitted/cnn_wind_speed.h5' 
    return load_model(path) 

