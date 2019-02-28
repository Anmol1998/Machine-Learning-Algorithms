# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 19:54:28 2019

@author: Anmol Agrawal
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering

data = pd.read_csv("C:/Users/Anmol Agrawal/Desktop/iris.csv")
print(data.head())

data=data.values[:,:-1]
print(data)

plt.figure(figsize=(10,7))
plt.title("Iris Dataset")
dend = shc.dendrogram(shc.linkage(data,method='ward'))

cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
cluster.fit_predict(data)

plt.figure(figsize=(10,7))
plt.scatter(data[:,0], data[:,1], c=cluster.labels_, cmap='rainbow')
