#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time Series Forecasting: SARIMA 
"""

from dspML import data, plot, utils 
from dspML.preprocessing import sequence 
from dspML.models.sequence import arima 
from dspML.evaluation import ForecastEval 

import pandas as pd 
import matplotlib.pyplot as plt 
import statsmodels.tsa.api as sm 

#%%

''' Load and Explore Signal '''

signal = data.Climate.humidity() 
plot.time_series_forecast(signal, title='Daily Humidity Time Series') 

#%%

''' Prepare Series '''

# split signal use observations from 2017 as test data 
fc_hzn = 24 
y_train, y_test = sequence.temporal_split(signal, fc_hzn) 

plot.time_series_forecast(y_train, title='Daily Humidity Train Data') 
plot.p_acf(y_train, lags=400) 
utils.ADF_test(y_train) 

#%%

''' Time Series Decomposition with Moving Averages '''

decomp = sm.seasonal_decompose(y_train, model='multiplicative', period=365) 
plt.rcParams.update({'figure.figsize':(14, 8)}) 
decomp.plot() 
plt.show() 

#%%

''' Define Model '''

p = (1,0,0,1) 
d = 0 
q = (0,0,0,1,1,0,1,0,1) 

model = sm.SARIMAX(y_train, 
                   order=(p, d, q), 
                   seasonal_order=(0, 0, 0, 0), 
                   enforce_stationarity=False, 
                   enforce_invertibility=False) 

fit = model.fit() 
print(fit.summary()) 

fit.plot_diagnostics(figsize=(14, 12)) 

#%%

y_pred = fit.forecast(steps=24) 
fc_eval = ForecastEval(y_test, y_pred) 
fc_eval.mse() 

# plot forecast 
plot.time_series_forecast(signal=y_train.iloc[-200:], 
                          signal_test=y_test, 
                          p_forecast=y_pred, 
                          title='Predicted Humidity Forecast') 



























