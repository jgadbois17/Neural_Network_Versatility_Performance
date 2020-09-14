#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dspML import plot 
from dspML.evaluation import ForecastEval 

import itertools 
import statsmodels.tsa.api as sm 

def AutoARIMA(x): 
    p = d = q = range(0, 2) 
    pdq = list(itertools.product(p, d, q)) 
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))] 
    order, seasonal_order, aic = [], [], [] 
    for param in pdq: 
        for season_param in seasonal_pdq: 
            try: 
                mod = sm.SARIMAX(x, order=param, seasonal_order=season_param, 
                                 enforce_stationarity=False, enforce_invertibility=False) 
                mod = mod.fit() 
                order.append(param) 
                seasonal_order.append(season_param) 
                aic.append(mod.aic) 
            except: 
                continue 
            print('SARIMAX{}x{} | AIC = {}'.format(param, season_param, mod.aic))  
    idx = aic.index(min(aic)) 
    model = sm.SARIMAX(x, order=order[idx], seasonal_order=seasonal_order[idx], 
                       enforce_stationarity=False, enforce_invertibility=False) 
    model = model.fit() 
    return model 

def ARIMA(x, order, seasonal_order): 
    model = sm.SARIMAX(x, order=order, seasonal_order=seasonal_order, 
                       enforce_stationarity=False, enforce_invertibility=False) 
    model = model.fit() 
    print(model.summary()) 
    print('AIC = {}'.format(model.aic)) 
    return model 

def validation_forecast(model, x_train): 
    y_fit = model.predict() 
    plot.forecast(x_train, y_fit) 
    ForecastEval(x_train, y_fit).mse() 

def predict_forecast(model, y, start='2017-03-24', end='2017-04-24', 
                     title='Predicted Forecast'): 
    fc = model.get_prediction(start=start, end=end).predicted_mean 
    plot.forecast(y, fc, title=title) 
    return fc 

