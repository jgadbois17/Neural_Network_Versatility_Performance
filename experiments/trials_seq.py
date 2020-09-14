#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dspML import data, plot, utils 
from dspML.preprocessing import sequence 
from dspML.models.sequence import arima 
from dspML.evaluation import ForecastEval 

import numpy as np 
import pandas as pd 
import itertools 
import matplotlib.pyplot as plt 
import statsmodels.tsa.api as sm 
from sktime.forecasting.model_selection import temporal_train_test_split 

import tensorflow as tf 
from tensorflow import nn 
from keras import Model, Sequential, layers, optimizers 
from keras.preprocessing import timeseries_dataset_from_array 

import tensorflow_probability as tfp 
tfpl = tfp.layers 
tfd = tfp.distributions 
tfb = tfp.bijectors 

#%%

''' Load Humidity Signal '''

# load signal 
signal = data.Climate.humidity() 
plot.signal_pd(signal, title='Daily Humidity Signal') 

# ACF/PACF 
plot.p_acf(signal, lags=50)  

#%%

''' Preprocess Data '''

# split data 
n = len(signal) 
signal_train = signal[0:int(0.9*n)] 
signal_test = signal[int(0.9*n):] 

# normalize data 
x_train, norm = sequence.normalize_train(signal_train) 
x_test = sequence.normalize_test(signal_test, norm) 

#%%

''' Data Windowing '''

train_data = timeseries_dataset_from_array(data=x_train, 
                                           targets=None, 
                                           sequence_length=32, 
                                           sequence_stride=1, 
                                           shuffle=True, 
                                           batch_size=32) 

test_data = timeseries_dataset_from_array(data=x_test, 
                                          targets=None, 
                                          sequence_length=32, 
                                          sequence_stride=1, 
                                          shuffle=True, 
                                          batch_size=32) 






















