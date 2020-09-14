#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn import metrics 
from sktime.performance_metrics.forecasting import smape_loss 
from dspML.models.image.model_utils import dice_coef, soft_dice 


''' Image Classification '''

class Classification: 
    def __init__(self, y_true, y_pred): 
        self.y_true = y_true 
        self.y_pred = y_pred 
    
    def accuracy(self): 
        print('Accuracy: {}'.format(metrics.accuracy_score(self.y_true, self.y_pred))) 
    
    def confusion_matrix(self): 
        print('Confusion Matrix:') 
        print(metrics.confusion_matrix(self.y_true, self.y_pred)) 
    
    def classification_report(self): 
        print('Classification Report:') 
        print(metrics.classification_report(self.y_true, self.y_pred))  


def compare_accuracy(y, lda_preds, cnn_preds): 
    lda_acc = metrics.accuracy_score(y, lda_preds) 
    cnn_acc = metrics.accuracy_score(y, cnn_preds) 
    print('LDA Test Accuracy = {}'.format(lda_acc)) 
    print('CNN Test Accuracy = {}'.format(cnn_acc)) 


''' Image Segmentation '''

def evaluate_unet(model, X_val, y_val, history): 
    scores = model.evaluate(X_val, y_val) 
    print(' ') 
    print('Validation Loss:', scores[0]) 
    print('Validation DICE:', scores[1]) 
    print(' ') 
    print('Plotting Loss and DICE Coefficient During Training:') 
    pd.DataFrame(history.history).plot(figsize=(10, 6)) 
    plt.grid(True) 
    plt.gca().set_ylim(0, 1) 
    plt.show() 

class Segmentation: 
    def __init__(self, y_true, y_pred): 
        self.y_true = y_true 
        self.y_pred = y_pred 
    
    def DICE(self): 
        dice = dice_coef(self.y_true, self.y_pred).numpy() 
        print('DICE Coefficient = {}'.format(dice)) 
    
    def soft_DICE(self): 
        sd_loss = soft_dice(self.y_true, self.y_pred).numpy() 
        print('Soft DICE Loss = {}'.format(sd_loss)) 

class SegResults: 
    def __init__(self, y_true, km_preds, unet_preds): 
        self.km = Segmentation(y_true, km_preds) 
        self.unet = Segmentation(y_true, unet_preds) 
    
    def DICE(self): 
        print('\nDICE Coefficients:') 
        print('K-Means = {}'.format(self.km.DICE()))  
        print('U-Net   = {}'.format(self.unet.DICE())) 
    
    def Soft_DICE(self): 
        print('\nSoft DICE Losses:') 
        print('K-Means = {}'.format(self.km.soft_DICE())) 
        print('U-Net   = {}'.format(self.unet.soft_DICE())) 

def DICE_Coefficients(y, km, unet): 
    dice_unet = dice_coef(y, unet) 
    dice_km = dice_coef(y, km) 
    print('\nDICE Coefficients:') 
    print('K-Means = {}'.format(dice_km.numpy())) 
    print('U-Net   = {}'.format(dice_unet.numpy())) 

def Soft_DICE_Loss(y, km, unet): 
    sd_unet = soft_dice(y, unet) 
    sd_km = soft_dice(y, km) 
    print('\nSoft DICE Losses:') 
    print('K-Means = {}'.format(sd_km.numpy())) 
    print('U-Net   = {}'.format(sd_unet.numpy())) 


''' Time Series Forecasting '''

class ForecastEval: 
    def __init__(self, y_true, y_fc): 
        self.y_true = y_true 
        self.y_fc = y_fc 
    
    def mse(self): 
        mse = metrics.mean_squared_error(self.y_true, self.y_fc) 
        print('MSE = {}'.format(round(mse, 2))) 
        print('RMSE = {}'.format(round(np.sqrt(mse), 2))) 
    
    def smape(self): 
        print('SMAPE Loss = {}'.format(smape_loss(self.y_true, self.y_fc))) 















