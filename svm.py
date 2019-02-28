# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 17:30:50 2019

@author: Anmol Agrawal
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns

sns.set(style="whitegrid",color_codes=True)

social_data=pd.read_csv("D:\Semester 6\ML\DA\Social_Network_Ads.csv")

print(social_data.head())

x = social_data.values[:,1:-1]
y = social_data.Purchased

for i in x:
    if(i[0]=='Male'):
        i[0]=0
    else:
        i[0]=1

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)


scalar = StandardScaler()
scalar.fit(x_train)
StandardScaler(copy=True, with_mean=True, with_std=True)

x_train = scalar.transform(x_train)
x_test = scalar.transform(x_test)


svclassifier = SVC(kernel='linear')
svclassifier.fit(x_train,y_train)

y_pred = svclassifier.predict(x_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test,y_pred))
print("Classification Report:")
print(classification_report(y_test,y_pred))
print("Accuracy Score:")
print(accuracy_score(y_test,y_pred)*100)

plot1 = sns.swarmplot(x ='Purchased', y='EstimatedSalary', hue='Gender', data = social_data)

