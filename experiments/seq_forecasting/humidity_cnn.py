#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time Series Forecasting 

Data: Daily Climate Humidity Signal 
Model: Recurrent Neural Network 
"""

from dspML import data, plot, utils 
from dspML.preprocessing import sequence 
from dspML.models.sequence import nnf 
from dspML.evaluation import ForecastEval 


''' Load Humidity Signal '''

# load signal 
signal = data.Climate.humidity() 
plot.signal_pd(signal, title='Daily Humidity Signal') 
utils.ADF_test(signal) 

''' Preprocess Signal '''

# split and normalize data 
fc_hzn = 7 
y, y_test = sequence.temporal_split(signal, fc_hzn) 
y, norm = sequence.normalize_train(y) 
y_test = sequence.normalize_test(y_test, norm) 

# create sequences 
time_steps = 10 
x_train, y_train = sequence.xy_sequences(y, time_steps) 

''' Convolutional Neural Network '''

# load model 
model = nnf.load_humidity(recurrent=False) 
model.summary() 

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

