# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 19:44:17 2019

@author: Anmol Agrawal
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data=pd.read_csv("C:/Users/Anmol Agrawal/Desktop/iris.csv")
print(data.head())
x = data.values[:,:-1]
y = data.values[:,-1]

print(x)
print(y)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1)

multiclassSVC = OneVsRestClassifier(SVC(kernel='linear',decision_function_shape='ovr'))
multiclassSVC.fit(x_train,y_train)

y_pred = multiclassSVC.predict(x_test)

print("Classification Report")
print(classification_report(y_test,y_pred))
print("Confusion Matrix")
print(confusion_matrix(y_test,y_pred))
print("Accuracy")
print(accuracy_score(y_test,y_pred))
