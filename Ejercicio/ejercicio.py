# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 13:36:28 2021

@author: Oscar P
"""

import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt

df = pd.read_csv('framingham.csv')


df.isnull().sum()


ax = plt.figure(figsize=(32,15)).add_subplot(111)
ax.imshow(df.isna().values.T)
ax.set_aspect(100)
plt.yticks(range(df.shape[1]), df.columns);

#sns.pairplot(df)


df_copy = df

df_copy.replace(np.nan, np.mean(df_copy), inplace = True)

df_copy.isnull().sum()

corr = df.astype('float64').corr()
ax = sns.heatmap(corr, annot = True)

x = df_copy.iloc[:,0:-1 ].values
y = df_copy.iloc[:, -1].values


from sklearn.model_selection import train_test_split as tts
x_train, x_test, y_train, y_test = tts(x, y, test_size= 0.25, random_state = 2)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)


from xgboost import XGBClassifier

classifier = XGBClassifier(num_class=1,
                                  learning_rate=0.1,
                                  num_iterations=500,
                                  max_depth=10,
                                  feature_fraction=0.7, 
                                  scale_pos_weight=1.5)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)


from sklearn.linear_model import LogisticRegression
classifier_1 = LogisticRegression().fit(x_train, y_train)
y_pred = classifier_1.predict(x_test)
accuracy_score(y_test, y_pred)

