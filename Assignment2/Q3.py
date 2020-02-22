#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 08:45:23 2019

@author: yuqi
"""
# k-means clustering

import matplotlib.pyplot as plt
import numpy
import sklearn.cluster as cluster
import sklearn.metrics as metrics
import pandas

CarsData = pandas.read_csv('cars.csv', delimiter=',')

# Get the number of rows
nCar = CarsData.shape[0]

# CarsData[['Horsepower', 'Weight']] Extract column Horsepower and Weight
trainData = numpy.reshape(numpy.asarray(CarsData[['Horsepower', 'Weight']]), (nCar, 2))

# Feature Scaling
# sc_Data = StandardScaler()
# trainData = sc_Data.fit_transform(trainData)

maxKClusters = 15

# Determine the number of clusters
nClusters = numpy.zeros(maxKClusters)
Elbow = numpy.zeros(maxKClusters)
Silhouette = numpy.zeros(maxKClusters)

for c in range(maxKClusters):
   KClusters = c + 1
   nClusters[c] = KClusters

   kmeans = cluster.KMeans(n_clusters=KClusters, random_state=60616).fit(trainData)

   # The Inertia value is the within cluster sum of squares deviation from the centroid
   # Inertia[c] = kmeans.inertia_
   
   if (1 < KClusters):
       Silhouette[c] = metrics.silhouette_score(trainData, kmeans.labels_)
   else:
       Silhouette[c] = numpy.NaN

   WCSS = numpy.zeros(KClusters)
   nC = numpy.zeros(KClusters)

   for i in range(nCar):   # nCar how man rows
      k = kmeans.labels_[i]
      nC[k] += 1
      diff1 = abs(trainData[i,0] - kmeans.cluster_centers_[k,0])
      diff2 = abs(trainData[i,1] - kmeans.cluster_centers_[k,1])
      WCSS[k] = WCSS[k] + (diff1*diff1+diff2*diff2)

   Elbow[c] = 0
   for k in range(KClusters):
      Elbow[c] += (WCSS[k] / nC[k])

print("|t 3a. The Elbow values and the Silhouette values for \
      1-cluster to 15-cluster solutions:")
print("N Clusters\t Elbow Value\t Silhouette Value:")
for c in range(maxKClusters):
   print('{:.0f} \t\t {:.4f} \t {:.4f}'
         .format(nClusters[c], Elbow[c], Silhouette[c]))
   
plt.plot(nClusters, Elbow, linewidth = 2, marker = 'o')
plt.grid(True)
plt.xlabel("Number of Clusters")
plt.ylabel("Elbow Value")
plt.xticks(numpy.arange(0, 16, step = 1))
plt.show()

plt.plot(nClusters, Silhouette, linewidth = 2, marker = 'o')
plt.grid(True)
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Value")
plt.xticks(numpy.arange(0, 16, step = 1))
plt.show()    

slop = numpy.zeros(maxKClusters)
acceleration = numpy.zeros(maxKClusters)

for c in range(maxKClusters-1):
    index = c + 1
    slop[index] =  Elbow[index] - Elbow[c]
    if (index >= 2):
        acceleration[index] = slop[index] - slop[c]
        
print("|t 3b. Based on the Elbow values, the Silhouette values and biggest \
      acceleration value(182592.2342) suggest number of clusters is 13")
print("N Clusters\t Slop\t\t Acceleration:")
for c in range(maxKClusters):
   print('{:.0f} \t\t {:.4f} \t {:.4f}'
         .format(nClusters[c], slop[c], acceleration[c]))
