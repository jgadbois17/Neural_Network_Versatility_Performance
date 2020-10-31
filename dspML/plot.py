#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
import matplotlib.pyplot as plt 
import statsmodels.tsa.api as sm 

''' Sequence '''

def signal_pd(x, title='Signal', figsize=(14,6), xlab='$t$', ylab='$X_t$', 
              rotation=0, yticks=None, thresh=None): 
    x.plot(figsize=figsize) 
    if thresh is not None: 
        plt.hlines(y=thresh, xmin=x.index[0], xmax=x.index[-1], 
                   color='red', label='anomaly threshold') 
    plt.title(title, size=16) 
    plt.xlabel(xlab), plt.xticks(rotation=rotation) 
    plt.ylabel(ylab), plt.yticks(ticks=yticks) 
    plt.grid(True) 
    plt.legend() 
    plt.show() 

def signal_np(x, title='Signal', figsize=(14,6), xlab='$t$', ylab='$X_t$', yticks=None): 
    plt.figure(figsize=figsize) 
    plt.plot(x) 
    plt.title(title, size=16) 
    plt.xlabel(xlab), plt.ylabel(ylab) 
    plt.yticks(ticks=yticks) 
    plt.grid(True) 
    plt.show() 

def p_acf(x, lags, figsize=(14,7)): 
    fig = plt.figure(figsize=figsize) 
    ax1 = fig.add_subplot(211) 
    fig = sm.graphics.plot_acf(x, lags=lags, ax=ax1) 
    ax2 = fig.add_subplot(212) 
    fig = sm.graphics.plot_pacf(x, lags=lags, ax=ax2) 

def time_series(signal, signal_test=None, p_forecast=None, 
                title='Time Series', xlab='$t$', ylab='$X_t$'):
    plt.plot(signal, '.-', label='observed') 
    if signal_test is not None: 
        plt.plot(signal_test, 'bx-', markersize=10, label='true forecast') 
    if p_forecast is not None: 
        plt.plot(p_forecast, 'ro-', label='predicted forecast') 
    plt.title(title, size=16) 
    plt.xlabel(xlab), plt.ylabel(ylab) 

def time_series_forecast(signal, signal_test=None, p_forecast=None, 
                         title='Time Series Forecast', figsize=(14, 7), 
                         grid=True, loc=1): 
    plt.figure(figsize=figsize) 
    time_series(signal, signal_test, p_forecast, title) 
    plt.grid(True) 
    plt.legend(loc=loc) 
    plt.show() 

def anomalies(x, idx_anomalies, figsize=(14,6), title='Anomalous Indices', xlab='Time', 
              ylab='Machine Temperature', rotation=0): 
    plt.clf() 
    sub = x.iloc[idx_anomalies, :] 
    x = x.join(sub, rsuffix='_anom') 
    ax = x['value'].plot(figsize=figsize, label='signal') 
    x['value_anom'].plot(ax=ax, color='r', label='anomalies')  
    plt.title(title, size=16) 
    plt.xlabel(xlab, size=12), plt.ylabel(ylab, size=12) 
    plt.xticks(rotation=rotation), plt.legend() 
    plt.show() 

''' Images '''

def plot_img(X, cmap=None):  
    plt.imshow(X, cmap=cmap) 
    plt.axis('off') 

class kmnist: 
    def __init__(self, X_train, y_train, X_test, y_test): 
        self.X_train = X_train 
        self.X_test = X_test 
        self.y_train = y_train 
        self.y_test = y_test 
    
    def train_observations(self, cmap=None): 
        plt.figure(figsize=(10, 10)) 
        for i in range(9): 
            rng = np.random.randint(len(self.X_train)) 
            plt.subplot(3, 3, (i+1)) 
            plot_img(self.X_train[rng], cmap=cmap) 
            plt.title('Label {}'.format(self.y_train[rng]), size=16) 
        plt.tight_layout() 
    
    def all_classes(self, figsize=(12, 6), cmap=None): 
        plt.figure(figsize=figsize) 
        for i in range(10): 
            plt.subplot(2, 5, (i+1)) 
            idx = np.where(self.y_train == i)[0][0] 
            plot_img(self.X_train[idx], cmap=cmap) 
            plt.title('Class Label = {}'.format(self.y_train[idx]), size=16) 
        plt.tight_layout() 
    
    def class_predictions(self, preds, idx=None, figsize=(8, 8), cmap=None): 
        if idx is None: 
            idx = np.random.randint(low=0, high=len(preds)) 
        plt.figure(figsize=figsize) 
        plot_img(self.X_test[idx], cmap=cmap) 
        plt.title('Predicted Class = {} | True Class = {}'
                  .format(self.y_test[idx], preds[idx]), size=16) 
        plt.show() 
    
    def rand_class_predictions(self, preds, figsize=(10, 5), cmap='gray'): 
        plt.figure(figsize=figsize) 
        for i in range(3): 
            idx = np.random.randint(len(preds)) 
            plt.subplot(1, 3, (i+1)) 
            plot_img(self.X_test[idx], cmap=cmap) 
            plt.title('Predicted: {} | True {}'
                      .format(preds[idx], self.y_test[idx]), size=14) 
        plt.tight_layout()  

def plot_observed_segmentations(X, y, cmap='gray', figsize=(10, 8)): 
    plt.figure(figsize=figsize) 
    for i in range(3): 
        idx = np.random.randint(len(X)-1) 
        plt.subplot(2, 3, (i+1)) 
        plot_img(X[idx], cmap=None) 
        plt.title('Image {}'.format(idx), size=16) 
        plt.subplot(2, 3, (i+4)) 
        plot_img(np.squeeze(y[idx]), cmap=cmap) 
        plt.title('Mask {}'.format(idx), size=16) 
    plt.tight_layout() 

class Nuclei: 
    def __init__(self, X_train, y_train, X_test=None, y_test=None, test=None): 
        self.X_train = X_train 
        self.y_train = y_train 
        self.X_test = X_test 
        self.y_test = y_test 
        self.test = test 
    
    def observations(self, cmap='gray', figsize=(10,10)): 
        plt.figure(figsize=figsize) 
        for i in range(3): 
            idx = np.random.randint(len(self.X_train)-1) 
            plt.subplot(2, 3, (i+1)) 
            plot_img(self.X_train[idx], cmap=cmap) 
            plt.title('Image {}'.format(idx), size=16) 
            plt.subplot(2, 3, (i+4)) 
            plot_img(np.squeeze(self.y_train[idx]), cmap=cmap) 
            plt.title('Mask {}'.format(idx), size=16) 
        plt.tight_layout() 
    
    def predicted_mask(self, preds, idx=None, cmap='gray', figsize=(10, 4.5)): 
        if idx is None: 
            idx = np.random.randint(len(preds)-1) 
        if len(preds) == len(self.y_train): 
            X = self.X_train 
            y = self.y_train 
            data = 'Train'
        else: 
            X = self.X_test 
            y = self.y_test 
            data = 'Test'
        plt.figure(figsize=figsize) 
        plt.subplot(131) 
        plot_img(X[idx], cmap=None) 
        plt.title('Image', size=16) 
        plt.subplot(132) 
        plot_img(np.squeeze(y[idx]), cmap=cmap) 
        plt.title('True Mask', size=16) 
        plt.subplot(133) 
        plot_img(np.squeeze(preds[idx]), cmap=cmap) 
        plt.title('Predicted Mask', size=16) 
        plt.suptitle('{} Observation Number: {}'.format(data, idx), size=18) 
        plt.tight_layout() 
    
    def new_random_segmentations(self, preds, cmap='gray', figsize=(10, 7)): 
        plt.figure(figsize=figsize) 
        for i in range(3): 
            idx = np.random.randint(len(self.test)) 
            img = self.test[idx] 
            mask = np.squeeze(preds[idx]) 
            plt.subplot(2, 3, (i+1)) 
            plot_img(img) 
            plt.title('Image {}'.format(idx), size=16) 
            plt.subplot(2, 3, (i+4)) 
            plot_img(mask) 
            plt.title('Predicted Mask', size=16) 
        plt.tight_layout() 
    
    def new_segmentation(self, preds, idx=None, cmap='gray', figsize=(10, 4.5)): 
        if idx is None: 
            idx = np.random.randint(len(self.test)) 
        plt.figure(figsize=figsize) 
        plt.subplot(121) 
        plot_img(self.test[idx], cmap=None) 
        plt.title('Image', size=16) 
        plt.subplot(122) 
        plot_img(np.squeeze(preds[idx]), cmap=cmap) 
        plt.title('Predicted Segmentation', size=16) 
        plt.suptitle('Observation Number {}'.format(idx), size=18) 
        plt.show() 





