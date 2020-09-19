#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow Time Series Forecasting Tutorial 

Data Windowing Section 
"""

from dspML import data 
from dspML.preprocessing import sequence 

import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 

#%%

''' 
Load and Prepare Data 
''' 

# load time series data 
signal = data.Climate.humidity() 

# split into train and test data 
train, test = sequence.temporal_split(signal, fc_hzn=114) 

# normalize signal 
train, norm = sequence.normalize_train(train) 
test = sequence.normalize_test(test, norm) 

#%%

''' 
Data Windowing 

Class to generate time series sequences 
'''

class WindowGenerator(): 
    def __init__(self, input_width, label_width, shift, 
                 train_df=train, test_df=test, 
                 label_columns=None): 
        # store raw data 
        self.train_df = train_df 
        self.test_df = test_df 
        
        # label columns indices 
        self.label_columns = label_columns 
        if label_columns is not None: 
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)} 
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)} 
        
        # window parameters 
        self.input_width = input_width 
        self.label_width = label_width 
        self.shift = shift 
        self.total_window_size = input_width + shift 
        self.input_slice = slice(0, input_width) 
        self.input_indices = np.arange(self.total_window_size)[self.input_slice] 
        self.label_start = self.total_window_size - self.label_width 
        self.labels_slice = slice(self.label_start, None) 
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice] 
    
    def __repr__(self): 
        return '\n'.join([f'Total window size: {self.total_window_size}', 
                          f'Input indices: {self.input_indices}', 
                          f'Label indices: {self.label_indices}', 
                          f'Label column name(s): {self.label_columns}']) 
    
    def split_window(self, features): 
        inputs = features[:, self.input_slice, :] 
        labels = features[:, self.labels_slice, :] 
        if self.label_columns is not None: 
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns], 
                axis=-1) 
        
        # set shapes 
        inputs.set_shape([None, self.input_width, None]) 
        labels.set_shape([None, self.label_width, None]) 
        return inputs, labels 
    
    def make_dataset(self, data): 
        data = np.array(data, dtype=np.float32) 
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data, targets=None, sequence_length=self.total_window_size, 
            sequence_stride=1, shuffle=True, batch_size=32) 
        ds = ds.map(self.split_window) 
        return ds 
    
    @property 
    def train(self): 
        return self.make_dataset(self.train_df) 
    
    @property
    def test(self): 
        return self.make_dataset(self.test_df) 
    
    @property 
    def example(self): 
        result = getattr(self, '_example', None) 
        if result is None: 
            result = next(iter(self.train)) 
            self._example = result 
        return result 
    
    def plot(self, model=None, plot_col='humidity', max_subplots=3): 
        inputs, labels = self.example 
        plt.figure(figsize=(12, 8)) 
        plot_col_index = self.column_indices[plot_col] 
        max_n = min(max_subplots, len(inputs)) 
        for n in range(max_n): 
            plt.subplot(3, 1, n+1) 
            plt.ylabel(f'{plot_col} [normed]') 
            plt.plot(self.input_indices, inputs[n, :, plot_col_index], 
                     label='Inputs', marker='.', zorder=-10) 
            if self.label_columns: 
                label_col_index = self.label_columns_indices.get(plot_col, None) 
            else: 
                label_col_index = plot_col_index 
            if label_col_index is None: 
                continue 
            plt.scatter(self.label_indices, labels[n, :, label_col_index], 
                        edgecolors='k', label='Labels', c='#2ca02c', s=64) 
            if model is not None: 
                predictions = model(inputs) 
                plt.scatter(self.label_indices, predictions[n, :, label_col_index], 
                            marker='X', edgecolors='k', label='Prediction', 
                            c='#ff7f0e', s=64) 
            if n == 0: 
                plt.legend() 
        plt.xlabel('Time') 

#%%

''' 
Example Window Generator 
'''

w = WindowGenerator(input_width=24, label_width=1, shift=24, 
                    label_columns=['humidity']) 
print(w) 
w.plot() 

#%%

''' 
One-Input to One-Output Window Generator 
'''

w_single_step = WindowGenerator(input_width=1, label_width=1, shift=1, 
                                label_columns=['humidity']) 
print(w_single_step) 

#%%

'''
Wider One-to-One Window Generator 

Generates windows of 30 days of consecutive inputs and labels at a time. 
This is still a one-to-one generator like the w_single_step generator but the 
time axis acts like the batch axis. 
Each prediction will be made independently with no interaction between time steps. 
'''

w_wide = WindowGenerator(input_width=30, label_width=30, shift=1, 
                         label_columns=['humidity']) 
print(w_wide) 
w_wide.plot() 

#%%

'''
Multiple-Inputs to Single-Output 

Generates batches of 3 days of inputs for 1 day of labels. 
'''

CONV_WIDTH = 3
w_conv = WindowGenerator(input_width=CONV_WIDTH, label_width=1, shift=1, 
                         label_columns=['humidity']) 
print(w_conv) 
w_conv.plot() 

#%%
















