#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time Series Forecasting 

SARIMA 
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
plot.signal_pd(signal, title='Humidity Time Series Signal') 
plot.p_acf(signal, lags=40) 
utils.ADF_test(signal) 

#%%

''' Prepare Series '''

# split signal use observations from 2017 as test data 
fc_hzn = 24 
y_train, y_test = sequence.temporal_split(signal, fc_hzn) 

plot.signal_pd(y_train, title='Humidity Train Signal') 
plot.p_acf(y_train, lags=40) 
utils.ADF_test(y_train) 

#%%

''' Auto-SARIMA | order=(1, 1, 1) | seasonal_order=(0, 1, 1, 12) | AIC=10749.049 '''

model = arima.AutoARIMA(y_train) 
model.plot_diagnostics(figsize=(12, 8)) 

# model validation 
arima.validation_forecast(model, y_train) 

#%%

''' SARIMA(2, 1, 1)x(0, 1, 1, 12) | AIC=10742.362 '''

model = arima.ARIMA(y_train, order=(2,1,1), seasonal_order=(0,1,1,12)) 
model.plot_diagnostics(figsize=(12, 8)) 

# model validation 
arima.validation_forecast(model, y_train) 

#%%

def rolling_forecast(y, fc_hzn, params): 
    fc = [] 
    for t in range(fc_hzn): 
        model = sm.SARIMAX(y, order=params['order'], seasonal_order=params['seasonal_order']) 
        model = model.fit() 
        yhat = model.forecast()[0] 
        y = y.append(yhat) 
        fc.append(yhat) 
    return pd.Series(fc) 

def plot_forecast(y, fc): 
    ax = y.plot(figsize=(14, 6), label='observed') 
    fc.plot(ax=ax, label='forecast') 
    ax.set_xlabel('Time', size=13) 
    ax.set_ylabel('Values', size=13) 
    ax.set_title('Forecast', size=17) 
    plt.legend(), plt.show() 

# predict forecast 
params = {'order':(2, 1, 1), 'seasonal_order':(0, 1, 1, 12)} 
fc = rolling_forecast(y_train, fc_hzn, params) 
fc.index = y_test.index 

# plot forecast 
plot_forecast(signal.loc['2017-01-01':], fc) 










