# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 21:55:16 2021

@author: wrace
"""

### Library Import ###

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sea
from data_prep import customerScaled


### Assessing Number of Clusters with SSE Elbow Method ###

sse = {} # sum of squared error

# calculate sse for differing values of K where K = 1:10
for k in range(1,10):
    kmeans = KMeans(n_clusters = k, random_state = 42)
    kmeans.fit(customerScaled)
    sse[k] = kmeans.inertia_

# plot sse plot using matplotlib and seaborn    
plt.title('Scaled Customer Data SSE Elbow Plot')
plt.xlabel('k')
plt.ylabel('SSE')
sea.pointplot(x=list(sse.keys()), y=list(sse.values()))
plt.show()

# Based on this elbow plot/SSE values, 3 clusters would be appropriate
# since there is a linear trend developing from k3-k4


### Implementing K-Means ###

# modelling kmeans
model = KMeans(n_clusters = 3, random_state = 42)
model.fit(customerScaled)
model.labels_.shape

# adding cluster labels to dataset
customerCluster = customerScaled
customerCluster["Cluster"] = model.labels_

# export final dataset for plotting and analysis in R with ggplot2
customerCluster.to_csv("Data\\customerCluster.csv")