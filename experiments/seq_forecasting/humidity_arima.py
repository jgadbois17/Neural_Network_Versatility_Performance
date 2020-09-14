#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time Series Forecasting 

Data: Daily Climate Humidity Signal 
Model: Seasonal ARIMA 
"""

import matplotlib.pyplot as plt 
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
fc_hzn = 10 
y_train, y_test = sequence.temporal_split(y, fc_hzn) 

# test stationarity 
utils.ADF_test(y) 

# plot ACF/PACF 
plot.p_acf(y, lags=365) 

#%%

''' Initial ARIMA Optimal Parameters '''

model = arima.AutoARIMA(y) 
model.plot_diagnostics(figsize=(14, 8)) 
plt.show() 

#%%

''' Seasonal ARIMA (2, 1, 4)x(2, 1, 4, 12) '''

model = arima.ARIMA(y_train, order=(2,1,4), seasonal_order=(2,1,4,12))
model.plot_diagnostics(figsize=(14, 8)) 
plt.show() 

# model validation 
arima.validation_forecast(model, y_train) 

#%%

# predicted forecast 
start = '2017-03-24' 
end = '2017-04-24' 
fc = arima.predict_forecast(model, y.loc['2017-02-01':'2017-04-24']) 
fc_eval = ForecastEval(y.loc[start:end], fc) 
fc_eval.mse() 





