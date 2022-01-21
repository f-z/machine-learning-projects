# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns; sns.set()
import csv

df = pd.read_csv('properties_2016.csv')
df.head(10)

# Preprocessing
df.dropna(axis=0, how='any',subset=['latitude','longitude'],inplace=True)

# Defining the x variable with lat and long coordinates
X=df.loc[:,['parcelid','latitude','longitude']]
X.head(10)

range_of_k_values = range(1, 10)
k_means = [KMeans(n_clusters=k) for k in range_of_k_values]

y_axis = df[['latitude']]
x_axis = df[['longitude']]
score = [k_means[i].fit(y_axis).score(y_axis) for i in range(len(k_means))]

# Visualizing for different values of k
plt.plot(range_of_k_values, score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()

k_means = KMeans(n_clusters=3, init='k-means++')
# Performing k-means clustering computation
k_means.fit(X[X.columns[1:3]])
X['cluster'] = k_means.fit_predict(X[X.columns[1:3]])
# Getting the centers of the fitted clusters
centers = k_means.cluster_centers_
# Getting the predicted clusters of each point
labels = k_means.predict(X[X.columns[1:3]])
X.head(10)

X.plot.scatter(x='latitude', y='longitude', c=labels, s=50, cmap='plasma')
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
