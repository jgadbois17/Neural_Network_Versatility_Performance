#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Classification Model 

Linear Discriminant Analysis 
"""

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 

def LDA(X_train, y_train, solver='svd'): 
    return LinearDiscriminantAnalysis(solver=solver).fit(X_train, y_train) 

