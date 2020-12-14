#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
from dspML import data, plot, utils 
from dspML.preprocessing import sequence 
from dspML.models.sequence import made 
from dspML.evaluation import Detection 

#%%

''' Load Signal '''

# load signal 
signal = data.machine_temperature() 
plot.signal_pd(signal, title='Machine Temperature Signal Anomaly Threshold', 
               yticks=np.arange(10, 110, 10), thresh=47) 

# train and test signal 
signal_train = signal.loc['2013-12-18 00:00:00':'2014-01-24 23:55:00'] 
signal_test = signal.loc['2014-02-01 00:00:00':] 

# extract true test anomalies and plot 
anoms_true = utils.extract_anomalous_indices(signal_test, thresh=47) 
plot.anomalies(signal_test, anoms_true) 

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
made.fit(model, x_train, epochs=1) 

# plot random sample 
made.plot_random_sample(dist, n=1000) 

# test probabilities 
px = dist.prob(x_test).numpy() 

#%%

# plot observations and their probabilities 
plot.signal_np(x_test, title='Normalized Test Signal') 
plot.signal_np(px, title='Probability of Signal Values') 

# let anomaly be observations with p(x) < 0.05 
anoms_pred = px < 0.05  

# evaluation 
met = Detection(anoms_true, anoms_pred) 
met.recall() 
met.precision() 
met.f1_score() 

# plot anomalies 
plot.anomalies(signal_test, anoms_pred) 

