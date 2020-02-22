#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 12:20:55 2019

@author: yuqi
"""
import matplotlib.pyplot as plt
import numpy
import sklearn.cluster as cluster
import sklearn.metrics as metrics
import pandas
import math

chicagoCompletedPotHoleData=pandas.read_csv('ChicagoCompletedPotHole.csv', delimiter=',')
nData = chicagoCompletedPotHoleData.shape[0]
chicagoCompletedPotHoleData['N_POTHOLES_FILLED_ON_BLOCK'] = chicagoCompletedPotHoleData['N_POTHOLES_FILLED_ON_BLOCK'].apply(lambda x: math.log(x))
chicagoCompletedPotHoleData['N_DAYS_FOR_COMPLETION'] = chicagoCompletedPotHoleData['N_DAYS_FOR_COMPLETION'].apply(lambda x: math.log(1 + x))

trainData = numpy.reshape(numpy.asarray(chicagoCompletedPotHoleData[['N_POTHOLES_FILLED_ON_BLOCK'
                                                                      ,'N_DAYS_FOR_COMPLETION'
                                                                      ,'LATITUDE', 'LONGITUDE']]), (nData, 4))

maxKClusters = 10

nClusters = numpy.zeros(maxKClusters)
Elbow = numpy.zeros(maxKClusters)
Silhouette = numpy.zeros(maxKClusters)

for c in range(1, maxKClusters):
   KClusters = c + 1
   nClusters[c] = KClusters

   kmeans = cluster.KMeans(n_clusters = KClusters, random_state = 20190327).fit(trainData)

   
   Silhouette[c] = metrics.silhouette_score(trainData, kmeans.labels_)
  

   WCSS = numpy.zeros(KClusters)
   nC = numpy.zeros(KClusters)

   for i in range(nData):   # nCar how man rows
      k = kmeans.labels_[i]
      nC[k] += 1
      diff1 = abs(trainData[i,0] - kmeans.cluster_centers_[k,0])
      diff2 = abs(trainData[i,1] - kmeans.cluster_centers_[k,1])
      diff3 = abs(trainData[i,2] - kmeans.cluster_centers_[k,2])
      diff4 = abs(trainData[i,3] - kmeans.cluster_centers_[k,3])
      WCSS[k] = WCSS[k] + (diff1*diff1+diff2*diff2+diff3*diff3+diff4*diff4)

   Elbow[c] = 0
   for k in range(KClusters):
      Elbow[c] += (WCSS[k] / nC[k])

print("The Elbow values and the Silhouette values for 1-cluster to 15-cluster solutions:")
print("N Clusters\t Elbow Value\t Silhouette Value:")
for c in range(maxKClusters):
   print('{:.0f} \t\t {:.4f} \t {:.4f}'
         .format(nClusters[c], Elbow[c], Silhouette[c]))


plt.plot(nClusters, Elbow, linewidth = 2, marker = 'o')
plt.grid(True)
plt.xlabel("Number of Clusters")
plt.ylabel("Elbow Value")
plt.xticks(numpy.arange(2, 11, step = 1))
plt.show()

plt.plot(nClusters, Silhouette, linewidth = 2, marker = 'o')
plt.grid(True)
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Value")
plt.xticks(numpy.arange(2, 11, step = 1))
plt.show()    


kmeans = cluster.KMeans(n_clusters = 4, random_state = 20190327).fit(trainData)
chicagoCompletedPotHoleData['KMeanCluster'] = kmeans.labels_

plt.scatter(chicagoCompletedPotHoleData[['LONGITUDE']],chicagoCompletedPotHoleData[['LATITUDE']] ,c = numpy.reshape(numpy.asarray(chicagoCompletedPotHoleData[['KMeanCluster']]), (nData, 1)), s=0.3)
plt.xlabel('LONGITUDE')
plt.ylabel('LATITUDE')
plt.grid(True)
#ax = plt.gca()
#ax.set_aspect('equal')
plt.show()

chicagoCompletedPotHoleDataNoTrans=pandas.read_csv('ChicagoCompletedPotHole.csv', delimiter=',')
trainData1 = chicagoCompletedPotHoleDataNoTrans[['N_POTHOLES_FILLED_ON_BLOCK', 'N_DAYS_FOR_COMPLETION', 'LATITUDE', 'LONGITUDE']].dropna()

from sklearn import tree
classTree = tree.DecisionTreeClassifier(criterion='gini', max_depth=2, random_state=20190327)
chicagoCompletedPotHole_DT = classTree.fit(trainData1, kmeans.labels_)
pred = chicagoCompletedPotHole_DT.predict_proba(trainData1)

print('Accuracy of Decision Tree classifier on training set: {:.6f}' .format(classTree.score(trainData1, kmeans.labels_)))
print('')

import graphviz
dot_data = tree.export_graphviz(chicagoCompletedPotHole_DT,
                                out_file=None,
                                impurity = True, filled = True,
                                feature_names = ['N_POTHOLES_FILLED_ON_BLOCK', 'N_DAYS_FOR_COMPLETION', 'LATITUDE', 'LONGITUDE'],
                                class_names = ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3'])

graph = graphviz.Source(dot_data)
graph

graph.render('decisiontree_outputc')

Ypro = pred
Y =kmeans.labels_
Y_pred = []
print(Ypro)
for i in range(len(Ypro)):
    if(Ypro[i][0]==1):
        Y_pred.append(0)
    if(Ypro[i][1]==1):
        Y_pred.append(1)
    if(Ypro[i][2]==1):
        Y_pred.append(2)
    if(Ypro[i][3]==1):
        Y_pred.append(3)
        
RASE = 0.0
for i in range(len(Y)):
    if (Y[i] == Y_pred[i]):
        RASE +=(1-1)**2
    else:
        RASE +=(0-1)**2
RASE = numpy.sqrt(RASE/len(Y))
print(RASE)
print("<Q11-e>")
