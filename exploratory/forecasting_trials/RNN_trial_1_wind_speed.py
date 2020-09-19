#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time Series Forecasting Humidity 

Recurrent Neural Network 
"""

from dspML import data, plot, utils 
from dspML.preprocessing import sequence 
from dspML.models.sequence import fc_rnn as rnn 
from dspML.evaluation import ForecastEval 

import numpy as np 
import pandas as pd 
from keras import Sequential, layers, optimizers 

#%%

''' Wind Speed Time Series Signal '''

signal = data.Climate.humidity() 
plot.signal_pd(signal, title='Humidity Time Series Signal') 
plot.p_acf(signal, lags=40) 
utils.ADF_test(signal) 

#%%

'''
Define Recurrent Neural Network 
'''

def Recurrent_Forecaster(input_shape, name=None): 
    model = Sequential(name=name) 
    model.add(layers.Input(shape=input_shape)) 
    
    model.add(layers.GRU(units=20, return_sequences=True)) 
    
    model.add(layers.TimeDistributed(layers.Dense(1))) 
    model.compile(optimizer=optimizers.Adam(), loss='mse') 
    return model 

# define model 
model = Recurrent_Forecaster(input_shape=(None, 1)) 
model.summary() 

#%%

''' Preprocessing Data '''

fc_hzn = 7 
time_steps = 30 
y, y_test = sequence.temporal_split(signal, fc_hzn) 
y, norm = sequence.normalize_train(y) 
y_test = sequence.normalize_test(y_test, norm) 
x_train, y_train = sequence.xy_sequences(y, time_steps) 


''' Fit Model and Predict Forecast '''

# fit model 
_= rnn.fit(model, x_train, y_train) 

# predict forecast 
y_pred = rnn.predict_forecast(model, x_train, steps=fc_hzn) 
y_pred.index = y_test.index 

# transform to original values 
y = sequence.to_original_values(y, norm) 
y_test = sequence.to_original_values(y_test, norm) 
y_pred = sequence.to_original_values(y_pred, norm) 

# evaluate model 
fc_eval = ForecastEval(y_test, y_pred) 
fc_eval.mse() 

# plot forecast 
plot.time_series_forecast(series=y.iloc[-100:], 
                          y=y_test, 
                          y_pred=y_pred, 
                          title='Wind Speed 7-Day Forecast') 











