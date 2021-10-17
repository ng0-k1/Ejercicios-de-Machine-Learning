# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 21:07:10 2021

@author: Oscar

"""
#Ejercicio de la información de salarios con regresión linear simple

import matplotlib.pyplot as plt
import pandas as pd 

dataset = pd.read_csv('Salary_Data.csv')

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 0 )

from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(x_train,y_train)

y_pred = regression.predict(x_test)

plt.scatter(x_train, y_train, color="red")
plt.plot(x_train, regression.predict(x_train), color="blue")
plt.title("Sueldo en relación con los años del conjunto de entrenamiento")
plt.xlabel("Años de experiencia")
plt.ylabel("Salarios")
plt.show()

plt.scatter(x_test, y_test, color="red")
plt.plot(x_train, regression.predict(x_train), color="blue")
plt.title("Sueldo en relación con los años del conjunto de entrenamiento")
plt.xlabel("Años de experiencia")
plt.ylabel("Salarios")
plt.show()