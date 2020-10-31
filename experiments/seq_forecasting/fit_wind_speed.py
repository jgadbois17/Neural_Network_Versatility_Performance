#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time Series Forecasting: Daily Wind Speed
Fitting Neural Networks 
"""

from dspML import data, plot 
from dspML.preprocessing import sequence 
from dspML.models.sequence import nnetfc as nnf 
from dspML.evaluation import ForecastEval 

# load and plot data 
signal = data.Climate.wind_speed() 
plot.time_series_forecast(signal, title='Daily Wind Speed Time Series') 

# split data 
fc_hzn = 7 
y, y_test = sequence.temporal_split(signal, fc_hzn) 

#%%

''' GRU Recurrent Network ''' 

# normalize series 
y, norm = sequence.normalize_train(y) 
y_test = sequence.normalize_test(y_test, norm) 

# create sequences 
time_steps = 30 
x_train, y_train = sequence.xy_sequences(y, time_steps) 

# define and fit model 
model = nnf.GRUNet() 
_= nnf.fit(model, x_train, y_train, path='GRU_wind_speed.h5', shuffle=True) 

# prediction 
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
                          title='Wind Speed 7-Day Forecast') 

#%%

''' Temporal Convolutional Network ''' 

# normalize series 
y, norm = sequence.normalize_train(y) 
y_test = sequence.normalize_test(y_test, norm) 

# create sequences 
time_steps = 10 
x_train, y_train = sequence.xy_sequences(y, time_steps) 

# define and fit model 
model = nnf.Convolutional(time_steps) 
_= nnf.fit(model, x_train, y_train, path='cnn_wind_speed.h5', shuffle=True) 

# prediction 
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
                          title='Wind Speed 7-Day Forecast') 

