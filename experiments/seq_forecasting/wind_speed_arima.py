#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time Series Forecasting 

Data: Daily Climate Wind Speed Signal 
Model: Seasonal ARIMA 
"""

import matplotlib.pyplot as plt 
from dspML import data, plot, utils 
from dspML.preprocessing import sequence 
from dspML.models.sequence import arima 
from dspML.evaluation import ForecastEval 

#%%

''' Load Wind Speed Signal '''

# load signal 
y = data.Climate.wind_speed() 
plot.signal_pd(y, title='Daily Wind Speed') 

# plot weekly moving average 
plot.signal_pd(y.rolling(window=7).mean(), title='Weekly Moving Average for Wind Speed') 

# train and test data 
fc_hzn = 10 
y_train, y_test = sequence.temporal_split(y, fc_hzn) 

# test stationarity 
utils.ADF_test(y) 

# plot ACF/PACF 
plot.p_acf(y, lags=365) 

#%%

''' Initial ARIMA Optimal Parameters: (1,1,1)x(0,1,1,12) | AIC = 8732.553 '''

model = arima.AutoARIMA(y) 
print(model.summary()) 
model.plot_diagnostics(figsize=(14, 8)) 
plt.show() 

#%%

''' Seasonal ARIMA (2,1,4)x(2,1,4,12) | AIC = 8465.672 '''

model = arima.ARIMA(y_train, order=(2, 1, 4), seasonal_order=(2, 1, 4, 12)) 
model.plot_diagnostics(figsize=(14, 8)) 
plt.show() 

# validate model 
arima.validation_forecast(model, y_train) 

#%%

# predicted forecast 
start = '2017-03-24' 
end = '2017-04-24' 
fc = arima.predict_forecast(model, y.loc['2017-02-01':'2017-04-24']) 
fc_eval = ForecastEval(y.loc[start:end], fc) 
fc_eval.mse() 

