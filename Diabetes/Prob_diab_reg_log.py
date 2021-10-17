# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 19:04:35 2021

@author: Oscar
"""

#Problema de diabetes solucionado con regresión logistica

import pandas as pd 
import numpy as np
dataset = pd.read_csv('diabetes2.csv')

x = dataset.iloc[:, 0:8].values
y = dataset.iloc[:, 8].values

from sklearn.linear_model import LinearRegression
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
X_opt = x[:, [0, 1, 2, 3, 4, 5, 6, 7]]
X_Modeled = backwardElimination(X_opt, SL)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_Modeled, y, random_state=0, test_size=0.2)


from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

from sklearn.linear_model import LogisticRegression
lg = LogisticRegression()
regressor = lg.fit(x_train, y_train)
y_pred = regressor.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.model_selection import cross_val_score
accuraccie = cross_val_score(estimator= regressor, X= x_train, y= y_train, cv=15)
accuraccie.mean()

accuracy2 = cross_val_score(estimator= regressor, X= x_test, y= y_test, cv=15)
accuracy2.mean()
