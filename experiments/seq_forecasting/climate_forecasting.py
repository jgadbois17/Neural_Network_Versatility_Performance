#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time Series Forecasting 

Data: 
    1. Forecasting Humidity 
    2. Forecasting Wind Speed 

Models: 
    Convolutional Networks 
    LSTM Networks 
    ARIMA Models 
"""

import matplotlib.pyplot as plt 
from dspML.data import data 
from dspML.sequential import utils_seq as utils 
from dspML.sequential import plot_seq as plot 
from dspML.sequential.models.forecast import optimal_arima 
from dspML.sequential.models.forecast import load_lstm_forecaster
from dspML.sequential.models.forecast import load_cnn_forecaster 

#%%

''' Load Climate Data '''

df_train = data.load_climate_train() 
df_test = data.load_climate_test() 
print(df_train.head()) 

#%%

''' 1. Forecasting Humidity '''

# plot daily humidity time series 
df_train['humidity'].plot(figsize=(10, 5)) 
plt.title('Daily Humidity', size=16) 
plt.show() 

# plot weekly humidity time series 
df_train['humidity'].resample('W').mean().plot(figsize=(10, 5)) 
plt.title('Weekly Humidity', size=16) 
plt.show() 

# plot monthly humidity time series 
df_train['humidity'].resample('M').mean().plot(figsize=(10, 5)) 
plt.title('Monthly Humidity', size=16) 
plt.show() 

#%%

''' Forecasting with ARIMA '''

ts_train = df_train['humidity'] 
ts_test = df_test['humidity'] 
ts = ts_train.append(ts_test) 

# test for stationarity 
utils.test_dickey_fuller(ts) 

# define model 
arima = optimal_arima(ts) 
print(arima.summary()) 

# get predicted forecast 
preds = arima.get_prediction(start=ts_test.index[0], dynamic=False).predicted_mean

# plot forecast 
plot.forecast(ts_test, preds[:-1]) 

# evaluate model 
arima_mse = utils.evaluate_model(ts_test, preds[:-1]) 

#%% 

''' Forecasting with LSTM '''

ts_train = utils.univar_ts(df_train['humidity']) 
ts_test = utils.univar_ts(df_test['humidity']) 

# preprocess data 
time_steps = 14 
x_train, y_train = utils.forecast_sequences(time_steps, ts_train) 

# load model 
lstm = load_lstm_forecaster() 
print(lstm.summary()) 

# get forecast 
lstm_forecast = utils.predict_forecast_nn(lstm, x_train, ts_test, time_steps) 

# plot forecast and evaluate model 
plot.forecast(ts_test, lstm_forecast) 
utils.evaluate_model(ts_test, lstm_forecast) 

#%%

''' Forecasting with CNN '''

ts_train = utils.univar_ts(df_train['humidity']) 
ts_test = utils.univar_ts(df_test['humidity']) 

# preprocess data 
time_steps = 14 
x_train, y_train = utils.forecast_sequences(time_steps, ts_train) 

# load model 
cnn = load_cnn_forecaster() 
print(cnn.summary()) 

# get forecast 
cnn_forecast = utils.predict_forecast_nn(cnn, x_train, ts_test, time_steps) 

# plot forecast and evaluate model 
plot.forecast(ts_test, cnn_forecast) 
utils.evaluate_model(ts_test, cnn_forecast) 

#%%

''' 2. Forecasting Wind Speed '''

# plot daily wind speed time series 
df_train['wind_speed'].plot(figsize=(10, 5)) 
plt.title('Daily Wind Speed', size=16) 
plt.show() 

# plot weekly wind speed time series 
df_train['wind_speed'].resample('W').mean().plot(figsize=(10, 5)) 
plt.title('Weekly Wind Speed', size=16) 
plt.show() 

# plot mondthly wind speed time series 
df_train['wind_speed'].resample('M').mean().plot(figsize=(10, 5)) 
plt.title('Monthly Wind Speed', size=16) 
plt.show() 

#%%

''' Forecasting with ARIMA '''

ts_train = df_train['wind_speed'] 
ts_test = df_test['wind_speed'] 
ts = ts_train.append(ts_test) 

# test for stationarity 
utils.test_dickey_fuller(ts) 

# define model 
arima = optimal_arima(ts) 
print(arima.summary()) 

# get predicted forecast 
preds = arima.get_prediction(start=ts_test.index[0], dynamic=False).predicted_mean

# plot forecast 
plot.forecast(ts_test, preds[:-1]) 

# evaluate model 
arima_mse = utils.evaluate_model(ts_test, preds[:-1]) 

#%%

''' Forecasting with LSTM '''

ts_train = utils.univar_ts(df_train['wind_speed']) 
ts_test = utils.univar_ts(df_test['wind_speed']) 

# preprocess data 
time_steps = 14 
x_train, y_train = utils.forecast_sequences(time_steps, ts_train) 

# load model 
lstm = load_lstm_forecaster(humid=False) 
print(lstm.summary()) 

# get forecast 
lstm_forecast = utils.predict_forecast_nn(lstm, x_train, ts_test, time_steps) 

# plot forecast and evaluate model 
plot.forecast(ts_test, lstm_forecast) 
utils.evaluate_model(ts_test, lstm_forecast) 

#%%

''' Forecasting with CNN '''

ts_train = utils.univar_ts(df_train['wind_speed']) 
ts_test = utils.univar_ts(df_test['wind_speed']) 

# preprocess data 
time_steps = 14 
x_train, y_train = utils.forecast_sequences(time_steps, ts_train) 

# load model 
cnn = load_cnn_forecaster(humid=False)  
print(cnn.summary()) 

# get forecast 
cnn_forecast = utils.predict_forecast_nn(cnn, x_train, ts_test, time_steps) 

# plot forecast and evaluate model 
plot.forecast(ts_test, cnn_forecast) 
utils.evaluate_model(ts_test, cnn_forecast) 


