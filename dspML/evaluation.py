#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
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
    
    def precision(self): 
        print('Precision: {}'.format(metrics.precision_score(
            self.y_true, self.y_pred, average='macro'))) 
    
    def recall(self): 
        print('Recall: {}'.format(metrics.recall_score(
            self.y_true, self.y_pred, average='macro'))) 
    
    def f1_score(self): 
        print('F1-Score: {}'.format(metrics.f1_score(
            self.y_true, self.y_pred, average='macro'))) 
    
    def confusion_matrix(self): 
        print('Confusion Matrix:') 
        print(metrics.confusion_matrix(self.y_true, self.y_pred)) 
    
    def classification_report(self): 
        print('Classification Report:') 
        print(metrics.classification_report(self.y_true, self.y_pred)) 

''' Image Segmentation '''

class SegmentationMetrics: 
    def __init__(self, X_train, y_train, X_test=None, y_test=None): 
        self.X_train = X_train 
        self.y_train = y_train 
        self.X_test = X_test 
        self.y_test = y_test 
    
    def select_data(self, preds): 
        if len(preds) == len(self.y_train): 
            X = self.X_train 
            y = self.y_train 
            data = 'Train'
        else: 
            X = self.X_test 
            y = self.y_test 
            data = 'Test'
        return X, y, data 
    
    def DiceCoefficient(self, preds): 
        X, y, data = self.select_data(preds) 
        dice = dice_coef(y, preds).numpy() 
        print('{} Data Dice Coefficient = {}'.format(data, dice)) 
    
    def softDice_loss(self, preds): 
        X, y, data = self.select_data(preds) 
        loss = soft_dice(y, preds).numpy() 
        print('{} Soft Dice Loss = {}'.format(data, loss)) 

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


''' Anomaly Detection '''

class Detection: 
    def __init__(self, anoms_true, anoms_pred): 
        self.anoms_true = anoms_true 
        self.anoms_pred = anoms_pred 
    
    def precision(self): 
        print('Precision: {}'.format(metrics.precision_score(
            self.anoms_true, self.anoms_pred, average='macro'))) 
    
    def recall(self): 
        print('Recall: {}'.format(metrics.recall_score(
            self.anoms_true, self.anoms_pred, average='macro'))) 
    
    def f1_score(self): 
        f1 = metrics.f1_score(self.anoms_true, self.anoms_pred) 
        print('F1-Score = {}'.format(np.around(f1, 4))) 




