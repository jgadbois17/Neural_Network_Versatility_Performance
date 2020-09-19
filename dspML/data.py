#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
import pandas as pd 
import tensorflow_datasets as tfds 
tfds.disable_progress_bar() 

''' Climate Data ''' 

class ClimateSplit: 
    def train(): 
        path = 'dspML/datasets/daily_climate/DailyDelhiClimateTrain.csv' 
        df = pd.read_csv(path) 
        df['date'] = pd.DatetimeIndex(df['date'], yearfirst=True, freq='D') 
        df.set_index('date', inplace=True) 
        return df 
    
    def test(): 
        path = 'dspML/datasets/daily_climate/DailyDelhiClimateTest.csv' 
        df = pd.read_csv(path) 
        df['date'] = pd.DatetimeIndex(df['date'], yearfirst=True, freq='D') 
        df.set_index('date', inplace=True) 
        return df 

class Climate: 
    def full(): 
        df = ClimateSplit.train() 
        df = df.append(ClimateSplit.test()) 
        return df 
    
    def humidity(): 
        df = ClimateSplit.train() 
        df = df.append(ClimateSplit.test()) 
        x = df['humidity'].resample('D').mean() 
        return x 
    
    def wind_speed(): 
        df = ClimateSplit.train() 
        df = df.append(ClimateSplit.test()) 
        x = df['wind_speed'].resample('D').mean() 
        return x 

''' Machine Temperature Data '''

def machine_temperature(path='dspML/datasets/anomaly/machine_temp_system_failure.csv'): 
    x = pd.read_csv(path) 
    x['timestamp'] = pd.to_datetime(x['timestamp'], yearfirst=True) 
    x.set_index('timestamp', inplace=True) 
    return x 

''' Nuclei Dataset '''

class Nuclei: 
    def train(path = 'dspML/datasets/nuclei/train_data.npz'): 
        data = np.load(path) 
        return data['images'], data['masks'] 
    
    def test(path = 'dspML/datasets/nuclei/test_data_1.npz'): 
        data = np.load(path) 
        return data['images'] 

''' KMNIST Dataset '''

def kmnist(): 
    train, test = tfds.load('kmnist', split=['train', 'test'], batch_size=-1, 
                            shuffle_files=True, as_supervised=True) 
    train, test = tfds.as_numpy(train), tfds.as_numpy(test) 
    X_train, y_train, X_test, y_test =train[0], train[1], test[0], test[1] 
    print('\nTraining data dimensions:') 
    print(X_train.shape, y_train.shape) 
    print('\nTesting data dimensions:') 
    print(X_test.shape, y_test.shape) 
    return np.squeeze(X_train), y_train, np.squeeze(X_test), y_test 






