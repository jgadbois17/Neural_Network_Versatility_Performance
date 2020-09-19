#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time Series Analysis 

Manipulating Time Series 
"""

import numpy as np 
import pandas as pd 
import tensorflow as tf 
import matplotlib.pyplot as plt 

from dspML import data, plot 
from dspML.preprocessing import sequence 


# load signal 
signal = data.Climate.humidity() 

# original signal 
plot.signal_pd(signal, title='Humidity Time Series Signal') 

# transformed signal 
plot.signal_pd(signal.diff(periods=1)) 



