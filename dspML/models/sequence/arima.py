#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dspML import plot 
from dspML.evaluation import ForecastEval 

import itertools 
import statsmodels.tsa.api as sm 

def AutoARIMA(x, seasons=None, verbose=True):
    p = d = q = range(0, 2) 
    pdq = list(itertools.product(p, d, q)) 
    if seasons is not None: 
        seasonal_pdq = [(x[0], x[1], x[2], seasons) for x in list(itertools.product(p, d, q))] 
    else: 
        seasonal_pdq = [(0, 0, 0, 0)] 
    order, seasonal_order, aic = [], [], [] 
    for param in pdq: 
        for season_param in seasonal_pdq: 
            try: 
                mod = sm.SARIMAX(x, order=param, seasonal_order=season_param) 
                mod = mod.fit() 
                order.append(param) 
                seasonal_order.append(season_param) 
                aic.append(mod.aic) 
            except: 
                continue 
            if verbose: 
                print('ARIMA{}x{} | AIC = {}'.format(param, season_param, mod.aic))  
    idx = aic.index(min(aic)) 
    model = sm.SARIMAX(x, order=order[idx], seasonal_order=seasonal_order[idx]) 
    model = model.fit() 
    print(model.summary()) 
    return model 

def ARIMA(x, order, seasonal_order=(0, 0, 0, 0)): 
    model = sm.SARIMAX(x, order=order, seasonal_order=seasonal_order) 
    model = model.fit() 
    print('AIC = {}'.format(model.aic)) 
    print(model.summary()) 
    return model 

def validation_forecast(model, x_train): 
    y_fit = model.predict() 
    plot.time_series_forecast(x_train, p_forecast=y_fit) 
    ForecastEval(x_train, y_fit).mse() 



