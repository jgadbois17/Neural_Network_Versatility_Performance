#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time Series Forecasting Wind Speed 

ARIMA Models 
"""

from dspML import data, plot, utils 
from dspML.preprocessing import sequence 
from dspML.models.sequence import arima 
from dspML.evaluation import ForecastEval 

#%%

''' Load and Explore Signal '''

signal = data.Climate.wind_speed() 
plot.signal_pd(signal, title='Wind Speed Time Series Signal') 
plot.p_acf(signal, lags=40) 
utils.ADF_test(signal) 

#%%

''' Prepare Signal '''

# split signal - save last available week for test 
fc_hzn = 7 
y_train, y_test = sequence.temporal_split(signal, fc_hzn) 

plot.signal_pd(y_train, title='Wind Speed Train Signal') 
plot.p_acf(y_train, lags=40) 
utils.ADF_test(y_train) 

#%%

''' Auto-ARIMA '''

model = arima.AutoARIMA(y_train) 
model.plot_diagnostics(figsize=(12, 8)) 

#%%

''' Predict 7-step forecast '''

# predict forecast 
y_pred = model.forecast(steps=7) 

print('\nCompare predictions with real future values:') 
print('\nForecast:\n{}'.format(y_pred)) 
print('\nTrue Values:\n{}'.format(y_test)) 
print(' ') 

# evaluation 
fc_eval = ForecastEval(y_test, y_pred) 
fc_eval.mse() 

#%%

''' Plot Forecast '''

plot.time_series_forecast(series=y_train.iloc[-100:], 
                          y=y_test, 
                          y_pred=y_pred, 
                          title='Wind Speed 7-Day Forecast') 





