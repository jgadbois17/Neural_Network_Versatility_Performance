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

''' Wind Speed Time Series '''

# load and plot signal 
y = data.Climate.wind_speed() 
plot.signal_pd(y, title='Daily Wind Speed Time Series') 

# plot ACF/PACF 
plot.p_acf(y, lags=40) 

#%%

''' First Difference '''

y1 = y.diff().iloc[1:]  
plot.signal_pd(y1, title='Daily Wind Speed - First Difference') 

# ACF/PACF 
plot.p_acf(y1, lags=40) 

# ADF test 
utils.ADF_test(y1) 

#%%

''' Preprocess Data '''

fc_hzn = 24 
y_train, y_test = sequence.temporal_split(y, fc_hzn) 

''' Define Model '''

p = (0) 
d = 1 
q = (3) 
model = arima.ARIMA(y_train, order=(p,d,q))  

#%%

''' Humidity Time Series '''

y = data.Climate.humidity() 
plot.signal_pd(y, title='Daily Humidity Time Series') 

# plot ACF/PACF 
plot.p_acf(y, lags=40) 

#%%

''' First Difference '''

y1 = y.diff().iloc[1:]  
plot.signal_pd(y1, title='Daily Humidity - First Difference') 

# ACF/PACF 
plot.p_acf(y1, lags=40) 

# ADF test 
utils.ADF_test(y1) 

#%%

''' Preprocess Data '''

fc_hzn = 24 
y_train, y_test = sequence.temporal_split(y, fc_hzn) 

''' Define Model '''

p = (1, 0, 1) 
d = 1 
q = (2) 
model = arima.ARIMA(y_train, order=(p,d,q))  























