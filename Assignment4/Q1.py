#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 12:09:56 2019

@author: yuqi
"""

import matplotlib.pyplot as plt
import numpy
import pandas
import sklearn.cluster as cluster
import sklearn.decomposition as decomposition
import sklearn.metrics as metrics

# Read data

ChicagoDiabetesData = pandas.read_csv('ChicagoDiabetes.csv',
                               delimiter=',')
# Feature variables
X = ChicagoDiabetesData[['Crude Rate 2000',
                         'Crude Rate 2001',
                         'Crude Rate 2002',
                         'Crude Rate 2003',
                         'Crude Rate 2004',
                         'Crude Rate 2005',
                         'Crude Rate 2006',
                         'Crude Rate 2007',
                         'Crude Rate 2008',
                         'Crude Rate 2009',
                         'Crude Rate 2010',
                         'Crude Rate 2011']]
# Drop missing value
X = X.dropna()

nObs = X.shape[0]
nVar = X.shape[1]
# Calculate the Correlations among the variables
XCorrelation = X.corr(method = 'pearson', min_periods = 1)
# Extract the Principal Components
_thisPCA = decomposition.PCA(n_components = nVar)
_thisPCA.fit(X)

cumsum_variance_ratio = numpy.cumsum(_thisPCA.explained_variance_ratio_)

print('|t 1a. The maximum number of principla components: {}'.format(_thisPCA.components_.shape[1]))
print('|t 1b. Plot the Explained Variances against their indices:')

plt.plot(_thisPCA.explained_variance_ratio_, marker = 'o')
plt.xlabel('Index')
plt.ylabel('Explained Variance Ratio')
plt.xticks(numpy.arange(0,nVar))
plt.axhline((1/nVar), color = 'r', linestyle = '--')
plt.grid(True)
plt.show()

print('|t 1c. To explain at least 95% of the total variance, the top two principal components should be selected.\n')
cumsum_variance_ratio = numpy.cumsum(_thisPCA.explained_variance_ratio_)
plt.plot(cumsum_variance_ratio, marker = 'o')
plt.xlabel('Index')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.xticks(numpy.arange(0,nVar))
plt.grid(True)
plt.show()

print('|t 1d. The cumulative explained variance ratio accounted by the top two major principal components: {}\n'.format(cumsum_variance_ratio[1]))

# Transform the data using the first two principal components
_thisPCA = decomposition.PCA(n_components = 2)
X_transformed = pandas.DataFrame(_thisPCA.fit_transform(X))

# Find clusters from the transformed data
maxNClusters = 10

nClusters = numpy.zeros(maxNClusters-1)
Elbow = numpy.zeros(maxNClusters-1)
Silhouette = numpy.zeros(maxNClusters-1)
TotalWCSS = numpy.zeros(maxNClusters-1)
Inertia = numpy.zeros(maxNClusters-1)

for c in range(maxNClusters-1):
   KClusters = c + 2
   nClusters[c] = KClusters

   kmeans = cluster.KMeans(n_clusters=KClusters, random_state=20190405).fit(X_transformed)

   # The Inertia value is the within cluster sum of squares deviation from the centroid
   Inertia[c] = kmeans.inertia_
   
   if (KClusters > 1):
       Silhouette[c] = metrics.silhouette_score(X_transformed, kmeans.labels_)
   else:
       Silhouette[c] = float('nan')

   WCSS = numpy.zeros(KClusters)
   nC = numpy.zeros(KClusters)

   for i in range(nObs):
      k = kmeans.labels_[i]
      nC[k] += 1
      diff = X_transformed.iloc[i,] - kmeans.cluster_centers_[k]
      WCSS[k] += diff.dot(diff)

   Elbow[c] = 0
   for k in range(KClusters):
      Elbow[c] += (WCSS[k] / nC[k])
      TotalWCSS[c] += WCSS[k]

print('|t 1e. Plot the Elbow and the Silhouette charts against the number of clusters:')
# Draw the Elbow and the Silhouette charts  
plt.plot(nClusters, Elbow, linewidth = 2, marker = 'o')
plt.grid(True)
plt.xlabel("Number of Clusters")
plt.ylabel("Elbow Value")
plt.xticks(numpy.arange(2, maxNClusters, 1))
plt.show()

plt.plot(nClusters, Silhouette, linewidth = 2, marker = 'o')
plt.grid(True)
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Value")
plt.xticks(numpy.arange(2, maxNClusters, 1))
plt.show()

print('|t 1f. 4 clusters will be chosen based on the charts in e)\n')

# Fit the 4 cluster solution
kmeans = cluster.KMeans(n_clusters=4, random_state=20190405).fit(X_transformed)
X_transformed['Cluster ID'] = kmeans.labels_
ChicagoDiabetesData['KMeanCluster'] = kmeans.labels_

community_cluster0 = []
community_cluster1 = []
community_cluster2 = []
community_cluster3 = []

for i in range(46):
    if(kmeans.labels_[i] == 0):
        community_cluster0.append(ChicagoDiabetesData.iloc[i, 1])
    if(kmeans.labels_[i] == 1):
        community_cluster1.append(ChicagoDiabetesData.iloc[i, 1])
    if(kmeans.labels_[i] == 2):
        community_cluster2.append(ChicagoDiabetesData.iloc[i, 1])
    if(kmeans.labels_[i] == 3):
        community_cluster3.append(ChicagoDiabetesData.iloc[i, 1])
        
print('|t 1g. The names of the communities in cluster0: ', community_cluster0)
print('The names of the communities in cluster1: ', community_cluster1)
print('The names of the communities in cluster2: ', community_cluster2)
print('The names of the communities in cluster3: ', community_cluster3)

cluster0_annual_population = []
cluster1_annual_population = []
cluster2_annual_population = []
cluster3_annual_population = []
total_annual_population = []

cluster0_annual_discharges = []
cluster1_annual_discharges = []
cluster2_annual_discharges = []
cluster3_annual_discharges = []
total_annual_discharges = []

for i in range(46):
    if(kmeans.labels_[i] == 0):
        for j in range(12):
            cluster0_annual_population.append(ChicagoDiabetesData.iloc[i, 2+2*j]
            /(ChicagoDiabetesData.iloc[i, 3+2*j]/10000))
            cluster0_annual_discharges.append(ChicagoDiabetesData.iloc[i, 2+2*j])
    if(kmeans.labels_[i] == 1):
        for j in range(12):
            cluster1_annual_population.append(ChicagoDiabetesData.iloc[i, 2+2*j]
            /(ChicagoDiabetesData.iloc[i, 3+2*j]/10000))
            cluster1_annual_discharges.append(ChicagoDiabetesData.iloc[i, 2+2*j])
    if(kmeans.labels_[i] == 2):
        for j in range(12):
            cluster2_annual_population.append(ChicagoDiabetesData.iloc[i, 2+2*j]
            /(ChicagoDiabetesData.iloc[i, 3+2*j]/10000))
            cluster2_annual_discharges.append(ChicagoDiabetesData.iloc[i, 2+2*j])
    if(kmeans.labels_[i] == 3):
        for j in range(12):
            cluster3_annual_population.append(ChicagoDiabetesData.iloc[i, 2+2*j]
            /(ChicagoDiabetesData.iloc[i, 3+2*j]/10000))
            cluster3_annual_discharges.append(ChicagoDiabetesData.iloc[i, 2+2*j])
    for j in range(12):
        total_annual_population.append(ChicagoDiabetesData.iloc[i, 2+2*j]/(ChicagoDiabetesData.iloc[i, 3+2*j]/10000))
        total_annual_discharges.append(ChicagoDiabetesData.iloc[i, 2+2*j])


cluster0_annual_population1 = numpy.zeros(12)
cluster1_annual_population1 = numpy.zeros(12)
cluster2_annual_population1 = numpy.zeros(12)
cluster3_annual_population1 = numpy.zeros(12)
total_annual_population1 = numpy.zeros(12)

cluster0_annual_discharges1 = numpy.zeros(12)
cluster1_annual_discharges1 = numpy.zeros(12)
cluster2_annual_discharges1 = numpy.zeros(12)
cluster3_annual_discharges1 = numpy.zeros(12)
total_annual_discharges1 = numpy.zeros(12)

for j in range(12):
     for i in range(18):
         cluster0_annual_population1[j] += cluster0_annual_population[j+i*12]
         cluster0_annual_discharges1[j] += cluster0_annual_discharges[j+i*12]

for j in range(12):
     for i in range(9):
         cluster1_annual_population1[j] += cluster1_annual_population[j+i*12]
         cluster1_annual_discharges1[j] += cluster1_annual_discharges[j+i*12]

for j in range(12):
     for i in range(5):
         cluster2_annual_population1[j] += cluster2_annual_population[j+i*12]
         cluster2_annual_discharges1[j] += cluster2_annual_discharges[j+i*12]

for j in range(12):
     for i in range(14):
         cluster3_annual_population1[j] += cluster3_annual_population[j+i*12]
         cluster3_annual_discharges1[j] += cluster3_annual_discharges[j+i*12]


for j in range(12):
    for i in range(46):
        total_annual_population1[j] += total_annual_population[j+i*12]
        total_annual_discharges1[j] += total_annual_discharges[j+i*12]
    
cluster0_annual_rate = numpy.zeros(12)
cluster1_annual_rate = numpy.zeros(12)
cluster2_annual_rate = numpy.zeros(12)
cluster3_annual_rate = numpy.zeros(12)
total_annual_rate = numpy.zeros(12)

for i in range(12):
    cluster0_annual_rate[i] = (cluster0_annual_discharges1[i]/
                        cluster0_annual_population1[i])*10000
    cluster1_annual_rate[i] = (cluster1_annual_discharges1[i]/
                        cluster1_annual_population1[i])*10000
    cluster2_annual_rate[i] = (cluster2_annual_discharges1[i]/
                        cluster2_annual_population1[i])*10000
    cluster3_annual_rate[i] = (cluster3_annual_discharges1[i]/
                        cluster3_annual_population1[i])*10000

for i in range(12):
    total_annual_rate[i] = (total_annual_discharges1[i]/
                        total_annual_population1[i])*10000
annual_rate_list = [total_annual_rate]
new_df = pandas.DataFrame(columns=['2000',
                               '2001',
                               '2002',
                               '2003',
                               '2004',
                               '2005',
                               '2006',
                               '2007',
                               '2008',
                               '2009',
                               '2010',
                               '2011'], data=annual_rate_list)
new_df = new_df.rename(index = {0 :'Chicago’s annual crude hospitalization rate'})
print('\n|t 1h. The Chicago’s annual crude hospitalization rates from 2000 to 2011:', new_df)
years = ['2000', '2001', '2002', '2003',
        '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011']
years = list(years)
print('\n|t 1i. Plot the crude hospitalization rates in each cluster against the years:')
plt.plot(years, cluster0_annual_rate, marker = 'o',
         color = 'blue', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.plot(years, cluster1_annual_rate, marker = 'o',
         color = 'red', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.plot(years, cluster2_annual_rate, marker = 'o',
         color = 'yellow', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.plot(years, cluster3_annual_rate, marker = 'o',
         color = 'black', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.plot(years, total_annual_rate, marker = 'o',
         color = 'orange', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.tight_layout()
ac = plt.gca()
ac.legend(['Cluster0', 'Cluster1', 'Cluster2', 'Cluster3', 'Chicago'])
plt.title("crude rate against year")
plt.grid(True)
plt.xlabel("year")
plt.ylabel("crude rate")
plt.show()