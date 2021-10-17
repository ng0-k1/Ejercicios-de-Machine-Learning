# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 13:35:11 2021

@author: Usuario
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

dataset = pd.read_csv('Fish.csv')

x = dataset.iloc[:,1:8].values
y = dataset.iloc[:, 0].values

from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

y = y.reshape(-1,1)

labelencoder_y = preprocessing.LabelEncoder()
y[:, 0] = labelencoder_y.fit_transform(y[:, 0])
ct = ColumnTransformer([('one_hot_encoder',OneHotEncoder(categories='auto'),[0])],
                       remainder ='passthrough')

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size =0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x,y)

y_pred = regressor.predict(x_test)





import statsmodels.api as sm
def backwardElimination(x, sl):    
    numVars = len(x[0])    
    for i in range(0, numVars):        
        regressor_OLS = sm.OLS(y, x.tolist()).fit()        
        maxVar = max(regressor_OLS.pvalues).astype(float)        
        if maxVar > sl:            
            for j in range(0, numVars - i):                
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):                    
                    x = np.delete(x, j, 1)    
    regressor_OLS.summary()    
    return x 

SL = 0.05
X_opt = x[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)