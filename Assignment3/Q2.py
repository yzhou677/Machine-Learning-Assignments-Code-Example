#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 20:39:31 2019

@author: yuqi
"""
# Multinomial Logistic Regression

import pandas
import math

import statsmodels.api as stats

purchaseData = pandas.read_csv('Purchase_Likelihood.csv', delimiter = ',')


# 2b. Produce count table
marginalCounts = list(purchaseData.groupby('A').size())

print("|t 2b. The marginal counts of the categories of the target variable A: ")
print("category0: {}".format(marginalCounts[0]))
print("category1: {}".format(marginalCounts[1]))
print("category2: {}".format(marginalCounts[2]))


# 2c. Calculate the likelihood estimates of the predicted probabilities πij, j= 1,2,3 of the Intercept-only model
likelihoodEstimates =  []
for i in range(3):
    likelihoodEstimates.append(marginalCounts[i]/(marginalCounts[0] + marginalCounts[1] +marginalCounts[2]))

print("|t 2c. The maximum likelihood estimates of the predicted probabilities πi1: {} {} {}".format(likelihoodEstimates[0], likelihoodEstimates[1], likelihoodEstimates[2]))

# 2d. Calculate the log-likelihood value of this Intercept-only model
logLikelihoodValue = 0
for i in range(3):
    logLikelihoodValue =logLikelihoodValue + marginalCounts[i]*(math.log(likelihoodEstimates[i]))
print("|t 2d. The log-likelihood value of this Intercept-only model: {}".format(logLikelihoodValue))


# 2e
likelihoodEIT =[]
likelihoodEIT.append(0)
for i in range(1, 3):
    likelihoodEIT.append(math.log(likelihoodEstimates[i]/likelihoodEstimates[0]))
print("|t 2e. The maximum likelihood estimates of the Intercept terms β_j0,j=1,2,3: {} {} {}".format(likelihoodEIT[0], likelihoodEIT[1], likelihoodEIT[2]))

# 2f. Create and display a contingency table where group_size, homeowner, 
# and married_couple are on the row dimension, and A is on the column dimension
predictorList = []
predictorList.append(purchaseData.group_size)
predictorList.append(purchaseData.homeowner)
predictorList.append(purchaseData.married_couple)

countTable = pandas.crosstab(index = predictorList, columns = purchaseData.A, margins = True, dropna = True)
x = countTable.drop('All', 1)
percentTable1 = countTable.div(x.sum(1), axis='index')*100
print("|t 2f. The contingency table:")
print(percentTable1)

# 2h
A = purchaseData['A'].astype('category')
y = A

groupSize = purchaseData[['group_size']].astype('category')
homeOwner = purchaseData[['homeowner']].astype('category')
marriedCouple = purchaseData[['married_couple']].astype('category')
X = pandas.get_dummies(groupSize)
X2 = pandas.get_dummies(homeOwner)
X3 = pandas.get_dummies(marriedCouple)
X = X.join(X2)
X = X.join(X3) 
X = stats.add_constant(X, prepend=True)

logit = stats.MNLogit(y, X)
thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
thisParameter = thisFit.params
params0= list(thisParameter[0].values)
params1= list(thisParameter[1].values)

Odd10=[]
Odd20=[]
for i in range(8):
    Odd10.append(math.exp(params0[0] + params0[i+1]))
    Odd20.append(math.exp(params1[0] + params1[i+1]))
print("|t 2h. Category0 of A is used by the MNLogit function as the reference category.")

print("The log-likelihood value of this model: \n", logit.loglike(thisParameter.values))
print("The number of parameters in the model:{}".format(thisParameter.shape[0]*thisParameter.shape[1]))

print("Model Parameter Estimates:\n", thisFit.params)

# 2i
odd_Qi = []
for i in range(4):
    i= i+1
    for j in range (2):
        j = j + 5
        for c in range(2):
            c = c + 7
            odd_Qi.append(math.exp(thisParameter.iloc[0, 0] + thisParameter.iloc[i, 0] + thisParameter.iloc[j, 0] + thisParameter.iloc[c, 0]))
odd_QiMax = max(odd_Qi)
print("|t 2i. The values of group_size, homeowner, and married_couple are: 2, 1, 1")
print("The maximum odd value: {}".format(odd_QiMax))


# 2j
resultJ = math.exp(thisParameter.iloc[3,1] - thisParameter.iloc[1,1])
print("|t 2j. The odds ratio for group_size = 3 versus group_size = 1, and A = 2 versus A = 0: {}"
      .format(resultJ))

# 2k
resultK = math.exp((thisParameter.iloc[1,1] -thisParameter.iloc[1,0]) - (thisParameter.iloc[3,1] -thisParameter.iloc[3,0]))
print("|t 2k. The odds ratio for group_size = 1 versus group_size = 3, and A = 2 versus A = 1: {}"
      .format(resultK))

