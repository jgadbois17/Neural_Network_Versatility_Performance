#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os 
import numpy as np 
from keras import Model, layers, optimizers 
from keras.models import load_model 
from keras.callbacks import EarlyStopping, ModelCheckpoint 
from dspML.models.image.model_utils import (
    dice_coef, soft_dice, conv_block, pool, merge) 

def uNet(input_shape, loss, metrics, depth=4, base_filt_pow=6, lr=1e-3, name=None, 
         activation='relu', kernel=(3,3), kernel_init='he_normal', pool_size=2): 
    inputs = layers.Input(shape=input_shape) 
    # encoder 
    filts = [] 
    c_layers, p_layers = [], [] 
    for i in range(depth): 
        if len(c_layers) == 0: 
            x_in = inputs 
        else: 
            x_in = p_layers[-1] 
        filts.append(2**(base_filt_pow+i)) 
        c_layers.append(conv_block(x_in, filts[i], kernel, activation, kernel_init)) 
        p_layers.append(pool(c_layers[-1], pool_size)) 
    # bottleneck 
    n_filt = 2**(base_filt_pow+depth) 
    m = conv_block(p_layers[-1], n_filt, kernel, activation, kernel_init) 
    # decoder 
    filts = np.flip(filts) 
    skips = np.flip(c_layers) 
    up_layers = [] 
    for i in range(depth): 
        if len(up_layers) == 0: 
            x_in = m 
        else: 
            x_in = up_layers[-1] 
        up_layers.append(merge(x_in, skips[i], filts[i])) 
        up_layers.append(conv_block(up_layers[-1], filts[i], kernel, activation, kernel_init)) 
    outputs = layers.Conv2D(1, (1,1), activation='sigmoid')(up_layers[-1]) 
    model = Model(inputs=[inputs], outputs=[outputs], name=name) 
    model.compile(optimizer=optimizers.Adam(learning_rate=lr), loss=loss, metrics=metrics) 
    return model 

def fit_model(model, path, X, y, batch_size=16, epochs=500, patience=15): 
    if not os.path.exists('models/'): 
        os.makedirs('models/') 
    save = 'models/{}'.format(path) 
    cb = [ModelCheckpoint(save, monitor='val_loss', mode='min', verbose=1, save_best_only=True), 
          EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)] 
    hist = model.fit(X, y, batch_size, epochs, callbacks=cb, validation_split=0.1, shuffle=True) 
    return hist 

def load_uNet(path='dspML/models/image/fitted/unet_fitted_model.h5'): 
    return load_model(path, custom_objects={'dice_coef':dice_coef, 'soft_dice':soft_dice}) 

def predict_segmentation_masks(model, X): 
    preds = model.predict(X) 
    return (preds > 0.5).astype(np.uint8) 

