#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd 
import tensorflow as tf 
import statsmodels.tsa.api as sm 


def ADF_test(x):
    print('\nAugmented Dickey-Fuller Test:\n') 
    df_test = sm.stattools.adfuller(x, autolag='AIC') 
    df_output = pd.Series(df_test[0:4], index=['Test Statistic', 'p-val', 'Num. Lags', 
                                               'Num. Observations'])
    for key, value in df_test[4].items():
        df_output['Critical Value (%s)'%key] = value 
    print(df_output) 
    p_val = df_test[1] 
    if p_val <= 0.01: 
        print('\nConclusion:\nThe p-val <= 0.01 thus, the series is stationary') 
    else: 
        print('\nConclusion\nThe p-val > 0.01 thus, the series is non-stationary') 


def fft2d_filter(img, keep_frac=0.1): 
    fft = tf.signal.fft2d(img) 
    fft2 = fft.numpy().copy() 
    r,c = fft2.shape[:2] 
    fft2[int(r*keep_frac):int(r*(1-keep_frac))] = 0 
    fft2[:, int(c*keep_frac):int(c*(1-keep_frac))] = 0 
    return tf.signal.ifft2d(fft2).numpy().real 




