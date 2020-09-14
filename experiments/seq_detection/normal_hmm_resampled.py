#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anomaly Detection 

Data: Machine Temperature Signal Resampled Every 30 Minutes 
Model: Normal Hidden Markov Model 
"""

from dspML import data, plot 
from dspML.preprocessing import sequence 
from dspML.models.sequence import NormalHMM as HMM 

#%%

''' Load Machine Temperature Signal '''

# load signal 
signal = data.machine_temperature().resample('30T').mean() 
signal = signal.loc['2014-02-01 00:00:00':] 
plot.signal_pd(signal, title='Machine Temperature Signal', rotation=25) 

# convert to numpy and reduce dimension 
x = sequence.reduce_dims(sequence.to_numpy(signal)) 

#%%

''' Hidden Markov Model '''

# define HMM 
hmm, prior = HMM.NormalHMM(x, n_states=3, p_change=1e-3) 

# fit HMM 
means = HMM.fit(hmm, prior, x) 

# posterior inference 
posterior_dists, ml_means, ml_states = HMM.posterior_inference(hmm, x) 
HMM.plot_inferred_means(x, means, ml_means, ml_states) 

#%%

''' Detect Anomalies '''

# detect anomalies 
anoms = HMM.anomalous_means(ml_means) 

# plot anomalies 
plot.anomalies(signal, anoms) 

