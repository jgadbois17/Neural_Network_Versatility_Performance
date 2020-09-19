#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
import pandas as pd 
import tensorflow as tf 
import matplotlib.pyplot as plt 
from keras import Sequential, layers, optimizers 

from dspML import data, plot 
from dspML.preprocessing import sequence 
from dspML.models.sequence import fc_rnn as rnn 
from dspML.evaluation import ForecastEval 

#%%

# load signal 
signal = data.Climate.humidity() 
plot.signal_pd(signal, title='Humidity Time Series Signal') 

#%%





    




















