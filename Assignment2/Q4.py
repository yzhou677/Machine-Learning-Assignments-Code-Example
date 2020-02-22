#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 14:16:23 2019

@author: yuqi
"""
import matplotlib.pyplot as plt
import numpy
import pandas

import sklearn.cluster as cluster

Spiral = pandas.read_csv('Spiral.csv', delimiter=',')

# Get the number of rows
nObs = Spiral.shape[0]

print("|t 4a. By visula inspection, there are 2 clusters")
plt.scatter(Spiral[['x']], Spiral[['y']])
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

trainData = Spiral[['x','y']]
kmeans = cluster.KMeans(n_clusters=2, random_state=60616).fit(trainData)

Spiral['KMeanCluster'] = kmeans.labels_

print("|t 4b. The scatterplot using the K-mean cluster identifier to control the color scheme\n")
plt.scatter(Spiral[['x']], Spiral[['y']], c = numpy.reshape(numpy.asarray(Spiral[['KMeanCluster']]), (nObs, 1)))
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

import math
import sklearn.neighbors


print("|t Three nearest neighbors will be used\n")
kNNSpec = sklearn.neighbors.NearestNeighbors(n_neighbors = 3, algorithm = 'brute', metric = 'euclidean')
nbrs = kNNSpec.fit(trainData)
d3, i3 = nbrs.kneighbors(trainData)
# Retrieve the distances among the observations
distObject = sklearn.neighbors.DistanceMetric.get_metric('euclidean')
distances = distObject.pairwise(trainData)

# Create the Adjacency and the Degree matrices
Adjacency = numpy.zeros((nObs, nObs))
Degree = numpy.zeros((nObs, nObs))

for i in range(nObs):
    for j in i3[i]:
        if (i <= j):
            Adjacency[i,j] = math.exp(- distances[i][j])
            Adjacency[j,i] = Adjacency[i,j]

for i in range(nObs):
    sum = 0
    for j in range(nObs):
        sum += Adjacency[i,j]
    Degree[i,i] = sum
        
Lmatrix = Degree - Adjacency

from numpy import linalg as LA
evals, evecs = LA.eigh(Lmatrix)

print("|t 4d. The sequence plot of the first nine eigenvalues:\n")
# Series plot of the smallest nine eigenvalues to determine the number of clusters
plt.scatter(numpy.arange(0,9,1), evals[0:9,])
plt.xlabel('Sequence')
plt.ylabel('Eigenvalue')
plt.show()
Z = evecs[:,[0,1]]

plt.scatter(Z[[0]], Z[[1]])
plt.xlabel('Z[0]')
plt.ylabel('Z[1]')
plt.show()

kmeans_spectral = cluster.KMeans(n_clusters=2, random_state=60616).fit(Z)

Spiral['SpectralCluster'] = kmeans_spectral.labels_

print("|t 4e. Apply the K-mean algorithm on the first two eigenvectors that correspond to the first two smallest eigenvalues. The regenerated scatterplot: ")
plt.scatter(Spiral[['x']], Spiral[['y']], c = numpy.reshape(numpy.asarray(Spiral[['SpectralCluster']]), (nObs, 1)))
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()


