#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf 
import keras.backend as K 
from keras.layers import Conv2D, Dropout, MaxPool2D 
from keras.layers import Conv2DTranspose, concatenate 

def dice_coef(y_true, y_pred, axis=(1, 2, 3), e=1e-5): 
    y_true = tf.cast(y_true, dtype=tf.float32) 
    y_pred = tf.cast(y_pred, dtype=tf.float32) 
    numerator = 2*K.sum(y_true*y_pred, axis=axis) + e 
    denominator = K.sum(y_true, axis=axis) + K.sum(y_pred, axis=axis) + e 
    return K.mean(numerator/denominator) 

def soft_dice(y_true, y_pred, axis=(1, 2, 3), e=1e-5): 
    y_true = tf.cast(y_true, dtype=tf.float32) 
    y_pred = tf.cast(y_pred, dtype=tf.float32) 
    numerator = 2*K.sum(y_true*y_pred, axis=axis) + e 
    denominator = K.sum(y_true**2, axis=axis) + K.sum(y_pred**2, axis=axis) + e 
    loss = 1 - K.mean(numerator/denominator) 
    return loss 

def conv_block(x_in, n_filt, kernel, activation, kernel_init, drop_size=0.2): 
    x = Conv2D(n_filt, kernel, padding='same', activation=activation, 
               kernel_initializer=kernel_init)(x_in) 
    x = Dropout(drop_size)(x) 
    x = Conv2D(n_filt, kernel, padding='same', activation=activation, 
               kernel_initializer=kernel_init)(x) 
    return x 

def pool(x_in, pool_size): 
    return MaxPool2D(pool_size, padding='same')(x_in) 

def tr_conv(x_in, n_filt, kernel=(2,2), strides=(2,2)): 
    return Conv2DTranspose(n_filt, kernel, strides, padding='same', activation='relu')(x_in) 

def merge(x_in, x_skip, n_filt): 
    x = tr_conv(x_in, n_filt) 
    return concatenate([x, x_skip]) 

