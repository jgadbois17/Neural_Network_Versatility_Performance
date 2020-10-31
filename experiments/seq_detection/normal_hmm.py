#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anomaly Detection 

Data: Machine Temperature Signal 
Model: Normal Hidden Markov Model 
"""

import numpy as np 
from dspML import data, plot, utils 
from dspML.preprocessing import sequence 
from dspML.models.sequence import NormalHMM as HMM 
from dspML.evaluation import f1_score 

#%%

''' Load Machine Temperature Signal '''

# load signal 
signal = data.machine_temperature() 
signal = signal.loc['2014-02-01 00:00:00':] 
plot.signal_pd(signal, title='Machine Temperature Signal Anomaly Threshold', 
               rotation=25, yticks=np.arange(10, 110, 10), thresh=47) 

# extract true test anomalies and plot 
anoms_true = utils.extract_anomalous_indices(signal, thresh=47) 
plot.anomalies(signal, anoms_true) 

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
anoms_pred = HMM.anomalous_means(ml_means) 

# f1 score 
f1_score(anoms_true, anoms_pred) 

# plot anomalies 
plot.anomalies(signal, anoms_pred) 

