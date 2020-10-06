#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time Series Forecasting: Daily Wind Speed Time Series 
"""

from dspML import data, plot, utils 
from dspML.preprocessing import sequence 
from dspML.models.sequence import arima, nnetfc 
from dspML.evaluation import ForecastEval 

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

#%%

''' Load Humidity Time Series '''

signal = data.Climate.wind_speed() 
plot.signal_pd(signal, title='Daily Wind Speed Signal') 

#%%

''' ARIMA '''

# train and test data 
fc_hzn = 7 
y_train, y_test = sequence.temporal_split(signal, fc_hzn) 

# define model 
arima_ = arima.ARIMA(y_train, order=(0, 1, 3)) 

# get predicted forecast 
fc_arima = arima_.forecast(steps=7) 

# plot forecast 
plot.time_series_forecast(signal=y_train.iloc[-100:], 
                          signal_test=y_test, 
                          p_forecast=fc_arima, 
                          title='Wind Speed ARIMA 7-Day Forecast') 

#%%

''' RNN '''

# split and normalize data 
fc_hzn = 7 
y, y_test = sequence.temporal_split(signal, fc_hzn) 
y, norm = sequence.normalize_train(y) 
y_test = sequence.normalize_test(y_test, norm) 

# create sequences 
time_steps = 5 
x_train, y_train = sequence.xy_sequences(y, time_steps) 

# load model 
rnn_ = nnetfc.load_humidity(recurrent=True) 

# get predicted forecast 
fc_rnn = nnetfc.predict_forecast(rnn_, x_train, steps=fc_hzn) 
fc_rnn.index = y_test.index 

# original value scale 
y = sequence.to_original_values(y, norm) 
y_test = sequence.to_original_values(y_test, norm) 
fc_rnn = sequence.to_original_values(fc_rnn, norm) 

# plot forecast 
plot.time_series_forecast(signal=y.iloc[-100:], 
                          signal_test=y_test, 
                          p_forecast=fc_rnn, 
                          title='Wind Speed RNN 7-Day Forecast') 

#%%

''' CNN '''

# split and normalize data 
fc_hzn = 7 
y, y_test = sequence.temporal_split(signal, fc_hzn) 
y, norm = sequence.normalize_train(y) 
y_test = sequence.normalize_test(y_test, norm) 

# create sequences 
time_steps = 5 
x_train, y_train = sequence.xy_sequences(y, time_steps) 

# load model 
cnn_ = nnetfc.load_humidity(recurrent=False) 

# get predicted forecast 
fc_cnn = nnetfc.predict_forecast(cnn_, x_train, steps=fc_hzn) 
fc_cnn.index = y_test.index 

# original value scale 
y = sequence.to_original_values(y, norm) 
y_test = sequence.to_original_values(y_test, norm) 
fc_cnn = sequence.to_original_values(fc_cnn, norm) 

# plot forecast 
plot.time_series_forecast(signal=y.iloc[-100:], 
                          signal_test=y_test, 
                          p_forecast=fc_cnn, 
                          title='Wind Speed CNN 7-Day Forecast') 

#%%

''' Plot all Predicted Forecasts '''

plt.figure(figsize=(14, 16)) 

plt.subplot(3, 1, 1) 
plot.time_series(signal=y.iloc[-50:], 
                 signal_test=y_test, 
                 p_forecast=fc_arima, 
                 title='Humidity ARIMA 7-Day Forecast') 
plt.grid(True) 
plt.subplot(3, 1, 2) 
plot.time_series(signal=y.iloc[-50:], 
                 signal_test=y_test, 
                 p_forecast=fc_rnn, 
                 title='Humidity RNN 7-Day Forecast') 
plt.grid(True) 
plt.subplot(3, 1, 3) 
plot.time_series(signal=y.iloc[-50:], 
                 signal_test=y_test, 
                 p_forecast=fc_cnn, 
                 title='Humidity CNN 7-Day Forecast') 
plt.grid(True) 
plt.savefig('fig5.5_wind_speed_forecasts.png') 
plt.show() 









