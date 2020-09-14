#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
import tensorflow_probability as tfp 
from keras.optimizers import Adam 
tfpl = tfp.layers 
tfd = tfp.distributions 
tfb = tfp.bijectors 


def NormalHMM(x, n_states, p_change, scale_dist=5., scale_prior=10.): 
    n = len(x) 
    # initial distribution 
    init_state_logits = np.zeros([n_states], dtype=np.float32) 
    z_init = tfd.Categorical(logits=init_state_logits) 
    # transition distribution 
    tpm = p_change / (n_states-1) * np.ones([n_states, n_states], dtype=np.float32) 
    np.fill_diagonal(tpm, 1-p_change) 
    z_trans = tfd.Categorical(probs=tpm) 
    # observation distribution 
    trainable_mean = tf.Variable(np.mean(x) + tf.random.normal([n_states]), name='means') 
    x_dist = tfd.Normal(loc=trainable_mean, scale=scale_dist) 
    # define HMM and prior 
    hmm = tfd.HiddenMarkovModel(z_init, z_trans, x_dist, num_steps=n) 
    prior = tfd.Normal(loc=np.mean(x), scale=scale_prior) 
    return hmm, prior 

def fit(hmm, prior, x, epochs=501, opt=Adam(learning_rate=0.1)): 
    trainable_mean = hmm.trainable_variables[0] 
    
    def log_prob(): 
        return (tf.reduce_sum(prior.log_prob(trainable_mean)) + hmm.log_prob(x)) 
    
    @tf.function(autograph=False) 
    def train_op():
        with tf.GradientTape() as tape: 
            neg_log_prob = -log_prob() 
        grads = tape.gradient(neg_log_prob, [trainable_mean])[0] 
        opt.apply_gradients([(grads, trainable_mean)]) 
        return neg_log_prob, trainable_mean 
    
    for epoch in range(epochs): 
        loss, means = [t.numpy() for t in train_op()] 
        if epoch % 20 == 0: 
            print('Epoch {}: log prob = {} | means = {}'
                  .format(epoch, -loss.round(3), means.round(3))) 
    return means 

def posterior_inference(hmm, x): 
    means = hmm.variables[0].numpy() 
    posterior_dists = hmm.posterior_marginals(x) 
    posterior_probs = posterior_dists.probs_parameter().numpy() 
    ml_states = np.argmax(posterior_probs, axis=1) 
    ml_means = means[ml_states] 
    return posterior_dists, ml_means, ml_states 

def plot_inferred_means(x, means, ml_means, ml_states, title='Inferred Latent Means'):
    fig = plt.figure(figsize=(14, 6)) 
    ax = fig.add_subplot(1, 1, 1) 
    ax.plot(ml_means, c='blue', lw=3, label='inferred mean') 
    ax.plot(x, c='black', alpha=0.3, label='observed values') 
    ax.set_title(title, size=16) 
    ax.set_xlabel('time') 
    ax.set_ylabel('latent mean') 
    ax.legend(loc=4) 

def anomalous_state(ml_states, state): 
    anoms = ml_states == np.unique(ml_states)[state] 
    return anoms 

def anomalous_means(ml_means): 
    anoms = ml_means == np.min(ml_means) 
    return anoms 


