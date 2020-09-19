#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time Series Forecasting 

Data: Daily Climate Humidity Signal 
Model: Seasonal ARIMA 
"""

from dspML import data, plot, utils 
from dspML.preprocessing import sequence 
from dspML.models.sequence import arima 
from dspML.evaluation import ForecastEval 

#%%

''' Load Humidity Signal '''

# load signal 
y = data.Climate.humidity() 
plot.signal_pd(y, title='Daily Humidity Signal') 

# plot weekly moving average 
plot.signal_pd(y.rolling(window=7).mean(), title='Humidity Weekly Moving Average') 

# train and test data 
fc_hzn = 7 
y_train, y_test = sequence.temporal_split(y, fc_hzn) 

# test stationarity 
utils.ADF_test(y) 

# plot ACF/PACF 
plot.p_acf(y, lags=40) 

#%%

''' ARIMA '''

model = arima.AutoARIMA(y_train, verbose=True) 
model.plot_diagnostics(figsize=(14, 8)) 

''' Predict Forecast '''

y_pred = model.forecast(steps=7) 
fc_eval = ForecastEval(y_test, y_pred) 
fc_eval.mse() 

# plot forecast 
plot.time_series_forecast(signal=y_train.iloc[-100:], 
                          signal_test=y_test, 
                          p_forecast=y_pred, 
                          title='Predicted Humidity Forecast') 



