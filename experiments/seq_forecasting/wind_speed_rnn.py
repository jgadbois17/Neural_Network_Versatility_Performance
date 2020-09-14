#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time Series Forecasting 

Data: Daily Climate Wind Speed Signal 
Model: Recurrent Neural Network 
"""

from dspML import data, plot 
from dspML.preprocessing import sequence 
from dspML.models.sequence import fc_rnn 
from dspML.evaluation import ForecastEval 


''' Load Humidity Signal '''

# load signal 
signal = data.Climate.humidity() 
plot.signal_pd(signal, title='Daily Humidity Signal') 

# ACF/PACF 
plot.p_acf(signal, lags=50)  


''' Preprocess Signal '''

# split data 
fc_hzn = 10 
signal_train, signal_test = sequence.temporal_split(signal, fc_hzn) 

# normalize data 
signal_train, norm = sequence.normalize_train(signal_train) 
signal_test = sequence.normalize_test(signal_test, norm) 

# create sequences 
time_steps = 32 
x_train, y_train = sequence.xy_sequences(signal_train, time_steps) 


''' Recurrent Neural Network '''



