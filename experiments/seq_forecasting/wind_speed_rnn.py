#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time Series Forecasting 
Data: Daily Climate Wind Speed Signal 
Model: Recurrent Neural Network 
"""

from dspML import data, plot 
from dspML.preprocessing import sequence 
from dspML.models.sequence import nnf 
from dspML.evaluation import ForecastEval 


''' Load Wind Speed Signal '''

# load signal 
signal = data.Climate.wind_speed() 
plot.signal_pd(signal, title='Daily Wind Speed Signal') 

# ACF/PACF 
plot.p_acf(signal, lags=50)  


''' Preprocess Signal '''

# split and normalize data 
fc_hzn = 7 
y, y_test = sequence.temporal_split(signal, fc_hzn) 

# normalize data 
y, norm = sequence.normalize_train(y) 
y_test = sequence.normalize_test(y_test, norm) 

# create sequences 
time_steps = 5 
x_train, y_train = sequence.xy_sequences(y, time_steps) 


''' Recurrent Neural Network '''

# define model 
model = nnf.GRUNet() 
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
                          title='Wind Speed 7-Day Forecast') 

