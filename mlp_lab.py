# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 16:51:22 2019

@author: Anmol Agrawal
"""

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

balance_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data',sep= ',', header= None)

print(balance_data.head())

x = balance_data.values[:,1:-1]
y = balance_data.values[:,0]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)

scalar = StandardScaler()
scalar.fit(x_train)
StandardScaler(copy=True, with_mean=True, with_std=True)

x_train = scalar.transform(x_train)
x_test = scalar.transform(x_test)

mlp = MLPClassifier(hidden_layer_sizes=(13,13,13), max_iter=500)
mlp.fit(x_train,y_train)

y_pred = mlp.predict(x_test)

print("Classification report:")
print(classification_report(y_test,y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test,y_pred))
print("Accuracy Score:",accuracy_score(y_test,y_pred)*100)

