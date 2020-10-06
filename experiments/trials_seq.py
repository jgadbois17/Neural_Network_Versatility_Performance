#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dspML import data, plot, utils 
from dspML.preprocessing import sequence 
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
























