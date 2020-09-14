#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anomaly Detection 

Data: 
    Machine Temperature System Failure 

Models: 
    Convolutional Autoencoder 
"""

import numpy as np 
import ruptures as rpt 

from dspML.data import data 
from dspML.sequential import plot_seq as plot 
from dspML.sequential import utils_seq as utils  
from dspML.sequential.models.detection import load_ConvAE 

#%%

''' Load Machine Temperature Data '''

# load signal 
signal = data.load_machine_temp() 
plot.signal_pd(signal, title='Machine Temperature System Signal') 

# training signal 
signal_train = signal.loc['2013-12-20 00:00:00':'2014-01-24 00:00:00'] 
plot.signal_pd(signal_train, title='Training Signal') 

# test signal 
signal_test = signal.loc['2014-02-01 00:00:00':] 
plot.signal_pd(signal_test, title='Testing Signal') 

#%%

''' Window-Based Change Point Detection '''

# signal for February 
x_feb = np.array(signal_test) 

# define detector 
detector = rpt.Window(width=100, model='l2').fit(x_feb) 

# predict change points and get anomaly indices 
cps = detector.predict(pen=1/2*np.log(len(x_feb))) 
idx_anoms_win = utils.cp_to_anom_idx(cps) 

# plot anomaly indices 
plot.anomalies(signal_test, idx_anoms_win) 

#%%

''' Convolutional Autoencoder '''

# normalize data 
x_train, norm = utils.normalize_train(signal_train) 
x_test = utils.normalize_test(signal_test, norm) 

# signal to sequences 
time_steps = 288 
x_train = utils.to_sequence(np.array(x_train), time_steps) 
x_test = utils.to_sequence(np.array(x_test), time_steps) 

# load model 
model = load_ConvAE() 
model.summary() 

# get threshold 
threshold = utils.anomaly_threshold(model, x_train) 

# predict test anomalies and plot 
anomalies = utils.predict_anomalies(model, x_test, threshold) 
idx_anoms_ae = utils.anomaly_ids(anomalies, x_test, time_steps) 
plot.anomalies(signal_test, idx_anoms_ae) 





