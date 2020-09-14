#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time Series Forecasting 

Data: Daily Climate Humidity Signal 
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

# define model 
model = fc_rnn.Recurrent_fc(input_shape=(None, 1), name='Recurrent_Forecaster') 
model.summary() 

# fit model 
_= fc_rnn.fit(model, x_train, y_train) 


''' Predict Forecast '''

# predict forecast 
fc = fc_rnn.predict_forecast(model, x_train, fc_hzn, time_steps) 

# plot forecast 
plot.forecast(sequence.to_numpy(signal_test), fc) 

# evaluate forecast 
fc_eval = ForecastEval(sequence.to_numpy(signal_test), fc) 
fc_eval.mse() 




