#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
from keras import Sequential, layers, optimizers 
from keras.callbacks import EarlyStopping 


def Recurrent_fc(input_shape, name=None): 
    model = Sequential(name=name) 
    model.add(layers.Input(shape=input_shape)) 
    
    model.add(layers.SimpleRNN(20, return_sequences=True))
    model.add(layers.SimpleRNN(10)) 
    model.add(layers.Dense(1)) 
    model.compile(optimizer=optimizers.Adam(), loss='mse') 
    return model 

def fit(model, x, y, batch_size=32, epochs=1000, patience=50): 
    cb = [EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)] 
    hist = model.fit(x, y, batch_size=batch_size, epochs=epochs, callbacks=cb, 
                     validation_split=0.1, shuffle=True) 
    return hist 

def predict_forecast(model, train, steps, time_steps): 
    forecast = [] 
    train = np.squeeze(train[-1]) 
    for i in range(steps): 
        x = np.expand_dims(train[-time_steps:], axis=(0, 2)) 
        pred = model.predict(x) 
        forecast.append(pred[0][0]) 
        train = np.append(train, pred[0][0]) 
    return np.array(forecast) 



