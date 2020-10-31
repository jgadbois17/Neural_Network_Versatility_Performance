#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time Series Forecasting Humidity 
Recurrent Neural Network 
"""

from dspML import data, plot, utils 
from dspML.preprocessing import sequence 
from dspML.models.sequence import nnetfc as nnf 
from dspML.evaluation import ForecastEval 

import numpy as np 
import pandas as pd 
from keras import Sequential, layers, optimizers 

#%%

''' Humidity Time Series Signal '''

signal = data.Climate.humidity() 
plot.signal_pd(signal, title='Humidity Time Series Signal') 

#%%

''' Preprocessing Data '''

# split and normalize data 
fc_hzn = 7 
y, y_test = sequence.temporal_split(signal, fc_hzn) 
y, norm = sequence.normalize_train(y) 
y_test = sequence.normalize_test(y_test, norm) 

# create sequences 
time_steps = 10 
x_train, y_train = sequence.xy_sequences(y, time_steps) 


''' Temporal Convolutional Network '''

# define model 
model = nnf.Convolutional(time_steps) 
model.summary() 

# fit model 
_= nnf.fit(model, x_train, y_train, shuffle=True) 


''' Predict Forecast '''

# predict forecast 
y_pred = nnf.predict_forecast(model, x_train, steps=fc_hzn) 
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

#%%

''' Trial: Loading Saved Model '''

# data 
signal = data.Climate.humidity() 
fc_hzn = 7 
y, y_test = sequence.temporal_split(signal, fc_hzn) 
y, norm = sequence.normalize_train(y) 
y_test = sequence.normalize_test(y_test, norm) 
time_steps = 10 
x_train, y_train = sequence.xy_sequences(y, time_steps) 

# load model 
trial = nnf.load_humidity(recurrent=False) 

# predict forecast 
y_pred = nnf.predict_forecast(model, x_train, steps=fc_hzn) 
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














