# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 21:13:56 2021

@author: Usuario
"""
#Predicción de datos para el proceso de insuficiencia cardiaca
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
dataset = pd.read_csv('heart_failure_clinical_records_dataset.csv')

x = dataset.iloc[:,0:12].values
y = dataset.iloc[:, -1].values
#eliminación hacia atras
import statsmodels.api as sm
def bacwardelimination(x, sl):
    num_vars = len(x[0])
    for i in range (0, num_vars):
        regressor_OLS = sm.OLS(y, x.tolist()).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar>sl:
            for j in range(0, num_vars-i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):                    
                    x = np.delete(x, j, 1)  
    regressor_OLS.summary()    
    return x 

SL = 0.05
X_opt = x[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
X_Modeled = bacwardelimination(X_opt, SL)
    
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_Modeled, y, test_size=0.2, random_state=2)


from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

from sklearn.decomposition import PCA
pda = PCA(n_components = 5)
x_train = pda.fit_transform(x_train)
x_test = pda.transform(x_test)

explained_variance = pda.explained_variance_ratio_

from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)


from sklearn.model_selection import cross_val_score
accuracy = cross_val_score(estimator = regressor, X= x_train, y = y_train , cv=10)
accuracy.mean()

accuracy2 = cross_val_score(estimator = regressor, X= x_test, y = y_test , cv=10)
accuracy2.mean()

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)

#Arboles de clasificación
from sklearn.ensemble import RandomForestClassifier
regression = RandomForestClassifier(n_estimators=200, random_state = 20)
regression.fit(x_train, y_train)

y_pred2 = regression.predict(x_test)

accuracy3 = cross_val_score(estimator = regression, X= x_train, y = y_train , cv=20)
accuracy3.mean()

accuracy4 = cross_val_score(estimator = regression, X= x_test, y = y_test , cv=10)
accuracy4.mean()
cm2 = confusion_matrix(y_test, y_pred2)

accuracy_score(y_test, y_pred, normalize = False)



