#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
import matplotlib.pyplot as plt 
import statsmodels.tsa.api as sm 

''' Sequence '''

def signal_pd(x, title='Signal', figsize=(14,6), xlab='$t$', ylab='$X_t$', 
              rotation=0, yticks=None): 
    x.plot(figsize=figsize) 
    plt.title(title, size=16) 
    plt.xlabel(xlab), plt.xticks(rotation=rotation) 
    plt.ylabel(ylab), plt.yticks(ticks=yticks) 
    plt.grid(True) 
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
    
    def class_means(self, figsize=(12,7)): 
        n_class = np.unique(self.y_train) 
        plt.figure(figsize=figsize) 
        for i in range(10): 
            idx = np.where(self.y_train == n_class[i]) 
            Xbar = np.mean(self.X_train[idx], axis=0) 
            plt.subplot(2, 5, (i+1)) 
            plt.hist(Xbar) 
            plt.title('Class {} Mean'.format(i), size=14) 
        plt.tight_layout() 

class Nuclei: 
    def __init__(self, X, y, test): 
        self.X = X 
        self.y = y 
        self.test = test 
    
    def observations(self, cmap=None, figsize=(10,10)): 
        plt.figure(figsize=figsize) 
        for i in range(3): 
            rng = np.random.randint(len(self.X)-1) 
            plt.subplot(2, 3, (i+1)) 
            plot_img(self.X[rng], cmap=cmap) 
            plt.title('Image {}'.format(rng), size=16) 
            plt.subplot(2, 3, (i+4)) 
            plot_img(np.squeeze(self.y[rng]), cmap=cmap) 
            plt.title('Mask {}'.format(rng), size=16) 
        plt.tight_layout() 
    
    def train_predictions(self, preds, cmap=None, figsize=(10,5)): 
        idx = np.random.randint(len(self.X)-1) 
        plt.figure(figsize=figsize) 
        plt.subplot(1, 3, 1) 
        plot_img(self.X[idx], cmap=cmap) 
        plt.title('Image', size=16) 
        plt.subplot(1, 3, 2) 
        plot_img(np.squeeze(self.y[idx]), cmap=cmap) 
        plt.title('Mask', size=16) 
        plt.subplot(1, 3, 3) 
        plot_img(np.squeeze(preds[idx]), cmap=cmap) 
        plt.title('Predicted Mask', size=16) 
        plt.suptitle('Observation Number: {}'.format(idx), size=18) 
        plt.tight_layout() 
    
    def test_pred_masks(self, preds, cmap=None, figsize=(10, 7)): 
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

def nuclei_test_preds_comparison(x, preds_u, preds_k, idx=[0, 1, 2]): 
    plt.figure(figsize=(10, 10)) 
    for i in range(3): 
        plt.subplot(3, 3, (i+1)) 
        plt.imshow(x[idx[i]]) 
        plt.axis('off') 
        plt.title('Image {}'.format(idx[i]), size=14) 
        plt.subplot(3, 3, (i+4)) 
        plt.imshow(np.squeeze(preds_u[idx[i]]), cmap='gray') 
        plt.axis('off') 
        plt.title('U-Net Predicted Mask', size=14) 
        plt.subplot(3, 3, (i+7)) 
        plt.imshow(np.squeeze(preds_k[idx[i]]), cmap='gray')  
        plt.axis('off') 
        plt.title('K-Means Predicted Mask', size=14) 
    plt.tight_layout() 




