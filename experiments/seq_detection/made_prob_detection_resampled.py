#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anomaly Detection 

Data: Machine Temperature Signal Resampled Every 30 Minutes 
Model: Masked Autoencoder for Density Estimation 
"""

import numpy as np 
from dspML import data, plot 
from dspML.preprocessing import sequence 
from dspML.models.sequence import made 

#%%

''' Load Signal '''

# load signal 
signal = data.machine_temperature().resample('30T').mean() 
plot.signal_pd(signal, title='Machine Temperature Signal', yticks=np.arange(10, 110, 10)) 

# train and test signal 
signal_train = signal.loc['2013-12-18 00:00:00':'2014-01-24 23:30:00'] 
signal_test = signal.loc['2014-02-01 00:00:00':] 

# convert signals to numpy 
x_train = sequence.to_numpy(signal_train) 
x_test = sequence.to_numpy(signal_test) 

# normalize signals 
x_train, norm = sequence.normalize_train(x_train) 
x_test = sequence.normalize_test(x_test, norm) 

# plot normalized signals 
plot.signal_np(x_train, title='Normalized Train Signal') 
plot.signal_np(x_test, title='Normalized Test Signal') 

#%%

''' MADE Autoregressive Network '''

# define model 
model, dist = made.MADE(params=2) 
model.summary() 

# fit model 
made.fit(model, x_train, epochs=100) 

# plot random sample 
made.plot_random_sample(dist, n=1000) 

# test probabilities 
px = dist.prob(x_test).numpy() 

# plot observations and their probabilities 
plot.signal_np(x_test, title='Normalized Test Signal') 
plot.signal_np(px, title='Probability of Signal Values') 

# let anomaly be observations with p(x) < 0.05 
anoms = px < 0.05  

# plot anomalies 
plot.anomalies(signal_test, anoms) 


