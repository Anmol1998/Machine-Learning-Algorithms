# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 21:27:57 2019

@author: Anmol Agrawal
"""

import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

data = pd.read_csv("C:/Users/Anmol Agrawal/Desktop/wine.csv")

x1 = data.iloc[:,:-1].values
#x2 = data.iloc[:,1].values
y_tmp = data.iloc[:,-1].values

x = []
y = []

for i in range(len(x1)):
    x.append(x1[i])
    y.append([y_tmp[i]])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=1)

LinReg = linear_model.LinearRegression()

LinReg.fit(x_train, y_train)
y_pred = LinReg.predict(x_test)

print("The Linear Regression Coefficients:", LinReg.coef_)
print("The Mean Squared Error:", round(mean_squared_error(y_test, y_pred), 2))
print("The R Squared Score:", round(r2_score(y_test, y_pred), 2))