#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dspML import data, plot 

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

#%%

# climate time series data 
x_humid = data.Climate.humidity() 
x_wind = data.Climate.wind_speed() 

plt.figure(figsize=(14, 10)) 
plt.subplot(2, 1, 1) 
plot.time_series(x_humid, title='Daily Humidity Time Series') 
plt.grid(True) 
plt.subplot(2, 1, 2) 
plot.time_series(x_wind, title='Daily Wind Speed Time Series') 
plt.grid(True) 
plt.savefig('daily_time_series_data.png') 
plt.show() 


