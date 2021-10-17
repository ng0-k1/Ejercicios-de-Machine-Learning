# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 17:03:05 2021

@author: Usuario
"""

import pandas as pd 
import seaborn as sns
import numpy as np

df = pd.read_csv('diabetes.csv')
#Reemplazando los valores que tienen "," por "."
df["bmi"] = df["bmi"].str.replace(",",".").astype(float)
df["waist_hip_ratio"] = df["waist_hip_ratio"].str.replace(",",".").astype(float)
df["chol_hdl_ratio"] = df["chol_hdl_ratio"].str.replace(",",".").astype(float)

# %time df = pd.get_dummies(df, columns = ['gender'], drop_first = True)

#sns.pairplot(df, hue= "gender")

#dividiendo el conjunto de X y Y
x = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

#Convirtiendo variables categoricas a numericas
from sklearn import preprocessing
labelenconder_x = preprocessing.LabelEncoder()
labelenconder_y = preprocessing.LabelEncoder()


x[:,5] = labelenconder_x.fit_transform(x[:,5])
y = labelenconder_y.fit_transform(y)

%time from sklearn.preprocessing import StandardScaler
%time sc_x = StandardScaler()
%time x = sc_x.fit_transform(x)

#Eliminación hacia atras
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
X_opt = x[:, 0:14]
X_Modeled = backwardElimination(X_opt, SL)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_Modeled, y, random_state=10, test_size=0.2)

#Normalizando las variables
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)


#Reduccion de variables
from sklearn.decomposition import PCA
pca = PCA(n_components = 4)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
explained_variance = pca.explained_variance_ratio_


#Regresion Logistica
from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

#Arboles de clasificación
from sklearn.ensemble import RandomForestClassifier
rlm = RandomForestClassifier(n_estimators = 120) 
rlm.fit(x_train, y_train)
y_pred2 = rlm.predict(x_test)

accuracy_score(y_test, y_pred2)


from sklearn.linear_model import LinearRegression
regressor2 = LinearRegression()
regressor2.fit(x_train, y_train)

y_pred3 = regressor.predict(x_test)
accuracy_score(y_test, y_pred3)

#Support Vector Classification
from sklearn.svm import SVC
svr = SVC(kernel = 'rbf')
svr.fit(x_train, y_train)
y_pred4 = svr.predict(x_test)

accuracy_score(y_test, y_pred4)

#XGBOOST
from xgboost import XGBClassifier
classifier = XGBClassifier(use_label_econder = False)
classifier.fit(x_train, y_train)

y_pred5 = classifier.predict(x_test)
accuracy_score(y_test, y_pred5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


from sklearn.model_selection import cross_val_score
cvl = cross_val_score(estimator= classifier, X = x_train, y = y_train, cv=15)
cvl.mean()
cvl.std()