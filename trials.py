#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dspML import data, plot, utils 
from dspML.preprocessing import sequence, image 
from dspML import evaluation as ev 
from dspML.models.sequence import made 
from dspML.models.sequence import NormalHMM as HMM 
from dspML.models.sequence import arima 

import itertools 
import numpy as np 
import pandas as pd 
import tensorflow as tf 
import matplotlib.pyplot as plt 
import statsmodels.tsa.api as sm 
import tensorflow_probability as tfp 

sts = tfp.sts 
tfpl = tfp.layers 
tfb = tfp.bijectors 
tfd = tfp.distributions 
from tensorflow import nn, signal 
from keras import Model, Sequential, layers, optimizers 
from keras.preprocessing import timeseries_dataset_from_array 
from sktime.forecasting.model_selection import temporal_train_test_split 

#%%

''' Load Wind Speed Signal '''

# load signal 
y = data.Climate.wind_speed() 
plot.signal_pd(y, title='Daily Wind Speed') 

# train and test data 
fc_hzn = 10 
y_train, y_test = sequence.temporal_split(y, fc_hzn) 


''' Define ARIMA Model '''

# define model 
model = arima.ARIMA(y_train, order=(2, 1, 4), seasonal_order=(2, 1, 4, 12)) 

# residual analysis 
model.plot_diagnostics(figsize=(14, 8)) 
plt.show() 

# validate model 
arima.validation_forecast(model, y_train) 

#%%

''' Predict Forecast '''













