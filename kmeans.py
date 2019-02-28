# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 21:47:58 2019

@author: Anmol Agrawal
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

#UCI repository iris dataset
data = pd.read_csv("iris.csv")
print(data.head())

data = data.values[:,:-1]
print(data)

kmeans = KMeans(n_clusters = 3, init= 'k-means++', max_iter=300, n_init=10, random_state=0)

cluster = kmeans.fit_predict(data)

plt.figure(figsize=(10,7))
plt.scatter(data[:,0],data[:,1], c=cluster, s=50, cmap='rainbow')
