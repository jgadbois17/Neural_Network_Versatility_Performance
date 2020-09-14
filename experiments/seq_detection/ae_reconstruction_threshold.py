#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anomaly Detection 

Data: Machine Temperature Signal 
Model: Convolutional Autoencoder 
"""

from dspML import data, plot 
from dspML.preprocessing import sequence 
from dspML.models.sequence import ae_threshold as AE 

from keras.callbacks import EarlyStopping 

#%%

''' Load Signal '''

# load signal 
signal = data.machine_temperature() 
signal = signal.resample('30T').mean() 
plot.signal_pd(signal, title='Machine Temperature Signal', rotation=25) 

# training signal 
signal_train = signal.loc['2013-12-20 00:00:00':'2014-01-24 23:30:00'] 
plot.signal_pd(signal_train, title='Training Signal') 

# test signal 
signal_test = signal.loc['2014-02-01 00:00:00':] 
plot.signal_pd(signal_test, title='Testing Signal') 

#%%

''' Preprocess Signal '''

# normalize signals 
x_train, norm = sequence.normalize_train(signal_train) 
x_test = sequence.normalize_test(signal_test, norm) 

# convert to numpy 
x_train = sequence.to_numpy(x_train) 
x_test = sequence.to_numpy(x_test) 

# create sequences 
time_steps = 32 
x_train = sequence.x_sequence(x_train, time_steps) 
x_test = sequence.x_sequence(x_test, time_steps) 

#%%

''' Convolutional Autoencoder '''

# define encoder 
ae = AE.ConvAE(input_shape=x_train.shape[1:]) 
ae.layers[0].summary() 
ae.layers[1].summary() 

# fit model 
cb = [EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)] 
ae.fit(x_train, x_train, epochs=500, callbacks=cb, validation_split=0.1) 

# get threshold 
threshold = AE.anomaly_threshold(ae, x_train) 

# predict anomalies and plot 
anomalies = AE.predict_anomalies(ae, x_test, threshold) 
anom_ids = AE.anomaly_ids(anomalies, x_test, time_steps) 
plot.anomalies(signal_test, anom_ids) 




