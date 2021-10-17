# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 17:16:35 2021

@author: Oscar P
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

dataset = pd.read_csv('California_Houses.csv')
dataset.info()

#sns.pairplot(dataset)

x = dataset.iloc[:, 1:14].values
y = dataset.iloc[:,0].values

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x = sc_x.fit_transform(x)
'''
import statsmodels.api as sm
def BacwardRegressor(x,sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x.tolist()).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars-i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x,j,1)
    regressor_OLS.summary()
    return x
sl = 0.06
x_opt = x[:, 0:13]
X_modeled = BacwardRegressor(x_opt, sl)
      '''      
         


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 5)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

from sklearn.decomposition import PCA
pca = PCA(n_components = 5)
x_train = pca.fit_transform(x_train)
x_test = pca.fit_transform(x_test)

explained_variance = pca.explained_variance_ratio_


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators= 400)
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

from sklearn.metrics import explained_variance_score
explained_variance_score(y_test, y_pred)



from xgboost import XGBRFRegressor
classifier = XGBRFRegressor()
classifier.fit(x_train, y_train)
y_pred2 = classifier.predict(x_test)
explained_variance_score(y_test, y_pred2)



from sklearn.linear_model import LinearRegression
regressor = LinearRegression(n_jobs = -1, )
regressor.fit(x_train, y_train)

y_pred3 = regressor.predict(x_test)
explained_variance_score(y_test, y_pred3)


from sklearn.metrics import r2_score
r2_score(y_test, y_pred)




