#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time Series Forecasting Humidity 

Recurrent Neural Network 
"""

from dspML import data, plot, utils 
from dspML.preprocessing import sequence 
from dspML.models.sequence import nnetfc 
from dspML.evaluation import ForecastEval 

import numpy as np 
import pandas as pd 
from keras import Sequential, layers, optimizers 

#%%

''' Humidity Time Series Signal '''

signal = data.Climate.humidity() 
plot.signal_pd(signal, title='Humidity Time Series Signal') 
plot.p_acf(signal, lags=40) 
utils.ADF_test(signal) 

#%%

''' 
Recurrent Neural Network 

1. 2xRNN + 1xDense | time_steps=10 | RMSE=7.52 
2. 2xLSTM + 1xDense | time_steps=50 | RMSE=6.5 
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


''' Preprocessing Data '''

# split and normalize data 
fc_hzn = 7 
y, y_test = sequence.temporal_split(signal, fc_hzn) 
y, norm = sequence.normalize_train(y) 
y_test = sequence.normalize_test(y_test, norm) 

# create sequences 
time_steps = 14 
x_train, y_train = sequence.xy_sequences(y, time_steps) 


''' Fit Model and Predict Forecast '''

# fit model 
_= nnetfc.fit(model, x_train, y_train) 

# predict forecast 
y_pred = nnetfc.predict_forecast(model, x_train, steps=fc_hzn) 
y_pred.index = y_test.index 

# transform to original values 
y = sequence.to_original_values(y, norm) 
y_test = sequence.to_original_values(y_test, norm) 
y_pred = sequence.to_original_values(y_pred, norm) 

# evaluate model 
fc_eval = ForecastEval(y_test, y_pred) 
fc_eval.mse() 

# plot forecast 
plot.time_series_forecast(signal=y.iloc[-100:], 
                          signal_test=y_test, 
                          p_forecast=y_pred, 
                          title='Humidity 7-Day Forecast') 





















