# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 17:04:58 2021

@author: Oscar P
"""
#Carga de librerias
import pandas as pd 
import seaborn as sns 
import numpy as np 
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

#Carga y limpieza de datos
dataset = pd.read_csv('call_metrics_dataset1.csv', delimiter = ';')
dataset.index = pd.to_datetime(dataset['date'])
del(dataset['date'])
dataset['avg_aht'] = dataset['avg_aht'].str.replace('.','')
dataset['avg_aht'] = dataset['avg_aht'].str.replace(",", '.').astype(float)

#Correlación
corr = dataset.astype('float64').corr()
ax = sns.heatmap(corr, annot = True, linewidths = 1)


x = dataset.iloc[:,0:-1].values
y = dataset.iloc[:, -1].values
#División en entrenamiento y test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train, x_test = sc_x.fit_transform(x_train), sc_x.transform(x_test)

#usando PCA para reducción devariables
from sklearn.decomposition import PCA
pca = PCA(n_components = 3)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)



explained_variance = pca.explained_variance_ratio_

#Probando modelos

#XGBOOST
from xgboost import XGBClassifier
classifier = XGBClassifier(use_label_encoder = False, eval_metric = 'mlogloss').fit(x_train, y_train)
y_pred = classifier.predict(x_test)
accuracy_score(y_test, y_pred)

#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
classifier2 = RandomForestClassifier(n_estimators= 10).fit(x_train, y_train)
y_pred2 = classifier2.predict(x_test)
accuracy_score(y_test, y_pred2)

#SUPPORT VECTOR CLASSIFIER
from sklearn.svm import SVC
classifier3 = SVC(gamma = 1).fit(x_train, y_train)
y_pred3 = classifier3.predict(x_test)
accuracy_score(y_test, y_pred3)

#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
classifier4 = LogisticRegression(max_iter = 200).fit(x_train, y_train)
y_pred4 = classifier4.predict(x_test)
accuracy_score(y_test, y_pred4)



#Curva de aprendizaje
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt



title = "Curva de aprendizaje"
cv = ShuffleSplit(n_splits = 10, test_size = 0.3)
plot_learning_curve(classifier4, title, x_train, y_train, ylim = (0.6, 1.0), cv = cv, n_jobs = 2)




