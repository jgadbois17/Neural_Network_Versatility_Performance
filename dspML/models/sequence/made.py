#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
import tensorflow_probability as tfp 
from keras import Model, layers, optimizers 
from keras.callbacks import EarlyStopping 
tfd = tfp.distributions 
tfb = tfp.bijectors 

def MADE(params, hidden_units=[10], event_shape=1): 
    made = tfb.AutoregressiveNetwork(params=params, hidden_units=hidden_units) 
    dist = tfd.TransformedDistribution(
        distribution=tfd.Normal(loc=0., scale=1.), 
        bijector=tfb.MaskedAutoregressiveFlow(made), 
        event_shape=[event_shape]) 
    x_in = layers.Input(shape=(event_shape,), dtype=tf.float32) 
    log_p = dist.log_prob(x_in) 
    model = Model(x_in, log_p, name='Masked_Autoencoder_Density_Estimation') 
    model.compile(optimizer=optimizers.Adam(), loss=lambda _,log_prob: -log_prob) 
    return model, dist 

def fit(model, x, batch_size=32, epochs=10): 
    cb = [EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)]
    model.fit(x, y=np.zeros((len(x), 0), dtype=np.float32), 
              batch_size=batch_size, epochs=epochs, steps_per_epoch=len(x)//batch_size, 
              shuffle=True, verbose=True, callbacks=cb)  

def plot_random_sample(dist, n): 
    x = dist.sample(n).numpy() 
    x = np.squeeze(x) 
    plt.figure(figsize=(12, 6)) 
    plt.subplot(1, 2, 1) 
    plt.hist(x) 
    plt.title('Histogram from Random Sample', size=16) 
    plt.subplot(1, 2, 2) 
    plt.plot(x) 
    plt.title('Signal from Random Sample', size=16) 
    plt.tight_layout() 

