#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time Series Forecasting Exploratory Data Analysis 
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

''' First Difference '''

# compute first difference 
x1 = x.diff() 

# plot first difference and ACF/PACF 
plot.signal_pd(x1, title='Daily Humidity - First Difference') 
plot.p_acf(x1, lags=400) 

