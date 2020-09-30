#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time Series Forecasting 
Exploratory Data Analysis 
"""

from dspML import data, plot, utils 
from dspML.preprocessing import sequence 

import numpy as np 
import pandas as pd 
import statsmodels.tsa.api as sm 
import matplotlib.pyplot as plt 

#%%

''' Humidity Time Series '''

# load data 
x = data.Climate.humidity() 

# plot time series and ACF/PACF 
plot.signal_pd(x, title='Daily Humidity Time Series') 
plot.p_acf(x, lags=400) 

#%%

# resample to average weekly values and plot 
plot.signal_pd(x.resample('W').mean(), title='Weekly Humidity Time Series') 
plot.p_acf(x.resample('W').mean(), lags=60) 

#%%

# resample to average monthly values and plot 
plot.signal_pd(x.resample('MS').mean(), title='Monthly Humidity Time Series') 
plot.p_acf(x.resample('MS').mean(), lags=24) 

#%%

decomp = sm.seasonal_decompose(x, model='multiplicative') 
trend = decomp.trend 
season = decomp.seasonal  
y = x - trend 


plot.signal_pd(x, title='Daily Humidity - Time Series') 
plot.signal_pd(trend, title='Daily Humidity - Estimated Trend Component') 
plot.signal_pd(season, title='Daily Humidity - Estimated Seasonal Component') 
plot.signal_pd(y, title='Daily Humidity - Removed Trend') 

#%%

fft = np.fft.rfft(x) 

plt.figure(figsize=(12, 7)) 
plt.plot(abs(fft)) 
plt.show() 


