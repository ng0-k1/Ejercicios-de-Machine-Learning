# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 20:06:47 2021

@author: Usuario
"""
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score, explained_variance_score
dataset = pd.read_csv('test_scores.csv')

sns.pairplot(dataset, hue = "gender")
dataset['school_setting'].unique()
dataset.drop(['classroom', 'student_id', 'school'], axis = 1, inplace= True )

x = dataset.iloc[:, 0:7].values
y = dataset.iloc[:, -1].values

from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

lbl_X = preprocessing.LabelEncoder()
x[:,0] = lbl_X.fit_transform(x[:,0])
x[:,1] = lbl_X.fit_transform(x[:,1])
x[:, 2] = lbl_X.fit_transform(x[:, 2])
x[:, 4] = lbl_X.fit_transform(x[:, 4])
x[:,5] = lbl_X.fit_transform(x[:,5])

ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'),[0])],
                        remainder = 'passthrough')
x = np.array(ct.fit_transform(x), dtype = float)

x = x[:, 1:]


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y , test_size= 0.2, random_state = 2)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)

from sklearn.svm import SVC
regressor = SVC(gamma = 8)
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)
r2_square = r2_score(y_test, y_pred)


explained_variance_score(y_test, y_pred)

y_pred.std()


from xgboost import XGBRFRegressor
regressor2 = XGBRFRegressor().fit(x_train, y_train)
y_pred2 = regressor2.predict(x_test)
explained_variance_score(y_test, y_pred2)
y_pred2.std()
