#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Classification Model 

Convolutional Neural Network 
"""

import numpy as np 
from keras import Sequential, layers 


def ConvNet(in_shape, n_classes, name=None): 
    model = Sequential(name=name) 
    model.add(layers.InputLayer(input_shape=in_shape)) 
    model.add(layers.Conv2D(32, (3,3), strides=1, padding='same', activation='relu')) 
    model.add(layers.MaxPooling2D(pool_size=(2,2), padding='same')) 
    model.add(layers.Conv2D(64, (3,3), strides=1, padding='same', activation='relu')) 
    model.add(layers.MaxPooling2D(pool_size=(2,2), padding='same')) 
    model.add(layers.Flatten()) 
    model.add(layers.Dense(n_classes, activation='softmax')) 
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 
    return model 

def predictions(model, X): 
    preds = model.predict(X) 
    return np.argmax(preds, axis=1) 

