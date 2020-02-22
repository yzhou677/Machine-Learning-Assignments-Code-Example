#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 11:17:46 2019

@author: yuqi
"""
import numpy
import pandas
from pandas import DataFrame
from itertools import combinations



CustomerData = pandas.read_csv('CustomerSurveyData.csv',
                       delimiter=',')
trainData = CustomerData[['CreditCard', 'JobCategory', 'CarOwnership']]
trainData = trainData[['CreditCard', 'JobCategory', 'CarOwnership']].replace(numpy.NaN, 'Missing')
trainData = trainData.dropna()

# Convert the CreditCard nominal variable into dummy variables
customer_creditCard = trainData[['CreditCard']].astype('category')
creditCard_inputs = pandas.get_dummies(customer_creditCard)

# Convert the JobCategory nominal variable into dummy variables
customer_jobCategory = trainData[['JobCategory']].astype('category')
jobCategory_inputs = pandas.get_dummies(customer_jobCategory)

def EntropyIntervalSplit (
   inData,          # input data frame (predictor in column 0 and target in column 1)
   leftList):          # left list

   dataTable = inData  
   for i in leftList:
       dataTable['LE_Split'] = (dataTable.iloc[:,0].isin(leftList))
   
   crossTable = pandas.crosstab(index = dataTable['LE_Split'], columns = dataTable.iloc[:,1], margins = True, dropna = True)   

   nRows = crossTable.shape[0]
   nColumns = crossTable.shape[1]
   
   tableEntropy = 0
   for iRow in range(nRows-1):
      rowEntropy = 0
      for iColumn in range(nColumns):
         proportion = crossTable.iloc[iRow,iColumn] / crossTable.iloc[iRow,(nColumns-1)]
         if (proportion > 0):
            rowEntropy -= proportion * numpy.log2(proportion)
      #print('Row = ', iRow, 'Entropy =', rowEntropy)
      #print(' ')
      tableEntropy += rowEntropy *  crossTable.iloc[iRow,(nColumns-1)]
   tableEntropy = tableEntropy /  crossTable.iloc[(nRows-1),(nColumns-1)]
  
   return(tableEntropy)
   



inDataCreditCard = trainData[['CreditCard', 'CarOwnership']]

inDataCreditCard = inDataCreditCard.replace('American Express', 1)
inDataCreditCard = inDataCreditCard.replace('Discover', 2)
inDataCreditCard = inDataCreditCard.replace('MasterCard', 3)
inDataCreditCard = inDataCreditCard.replace('Visa', 4)
inDataCreditCard = inDataCreditCard.replace('Others', 5)

listAll = [1, 2, 3, 4, 5]
EV = EntropyIntervalSplit(inDataCreditCard,listAll)
print("|t 1a. The Entropy metric for the root node:{}".format(EV))

print("|t 1b.{} possible binary-splits that can be generated from the CreditCard predictor."
      .format(2**4 - 1))

def entropyCalculate(inData, 
                     dataList, 
                     lengthN):
    entropyResultList = []
    combinationList1 = []
    combinationList2 = []
    for i in range(int(lengthN/2)):
        listData = list(combinations(dataList, i+1))
        for listE in listData:
            combinationList1.append(listE)
            listE = list(listE)
            listE2 = [x for x in dataList if x not in listE]
            combinationList2.append(tuple(listE2))
            EV = EntropyIntervalSplit(inData,listE)
            entropyResultList.append(EV)
            #print('Split Entropy = ', EV)
    df = DataFrame({'combination1': combinationList1,
                    'combination2':combinationList2,
                    'entropy': entropyResultList})
    return(df)
    

list1 = [1, 2, 3, 4, 5]
entropyResult1 = entropyCalculate(inDataCreditCard, list1, 5)

# change 1, 2, 3, 4, 5 back to American Express, Discover, MasterCard, Visa, Others
listCombination1 = []
for i in range(15):
    tupleE = entropyResult1.iloc[i,0]
    listE = list(tupleE)
    listE = ['American Express' if v == 1 else v for v in listE]
    listE = ['Discover' if v == 2 else v for v in listE]
    listE = ['MasterCard' if v == 3 else v for v in listE]
    listE = ['Visa' if v == 4 else v for v in listE]
    listE = ['Others' if v == 5 else v for v in listE]
    
    tupleE1 = entropyResult1.iloc[i,1]
    listE1 = list(tupleE1)
    listE1 = ['American Express' if v == 1 else v for v in listE1]
    listE1 = ['Discover' if v == 2 else v for v in listE1]
    listE1 = ['MasterCard' if v == 3 else v for v in listE1]
    listE1 = ['Visa' if v == 4 else v for v in listE1]
    listE1 = ['Others' if v == 5 else v for v in listE1]

    combinationResult = '('
    for j in listE:
        combinationResult += j + ', '
    combinationResult = combinationResult + ') , ('
    for j in listE1:
        combinationResult += j + ', '
    combinationResult = combinationResult + ')'
    listCombination1.append(combinationResult)

listEntropy1 = list(entropyResult1.iloc[:,2])

entropyResult1 = DataFrame({'combination': listCombination1,
                    'entropy': listEntropy1})
print("|t 1c. The table:")
print(entropyResult1)

print("|t 1d. The optimal split for the CreditCard predictor: {}"
      .format(listCombination1[listEntropy1.index(min(listEntropy1))]))


print("--------------------------------------------------------------")
#
inDataJobCategory = trainData[['JobCategory', 'CarOwnership']]

inDataJobCategory = inDataJobCategory.replace('Agriculture', 1)
inDataJobCategory = inDataJobCategory.replace('Crafts', 2)
inDataJobCategory = inDataJobCategory.replace('Labor', 3)
inDataJobCategory = inDataJobCategory.replace('Missing', 4)
inDataJobCategory = inDataJobCategory.replace('Professional', 5)
inDataJobCategory = inDataJobCategory.replace('Sales', 6)
inDataJobCategory = inDataJobCategory.replace('Service', 7)

list2 = [1, 2, 3, 4, 5, 6, 7]
entropyResult2 = entropyCalculate(inDataJobCategory, list2, 7)
print("|t 1e.{} possible binary-splits that can be generated from the JobCategory predictor."
      .format(2**6 - 1))


# change 1, 2, 3, 4, 5, 6, 7 back to
# Agriculture, Crafts, Labor, Missing, Professional, Sales, Service
listCombination2 = []
for i in range(63):
    tupleE = entropyResult2.iloc[i,0]
    listE = list(tupleE)
    listE = ['Agriculture' if v == 1 else v for v in listE]
    listE = ['Crafts' if v == 2 else v for v in listE]
    listE = ['Labor' if v == 3 else v for v in listE]
    listE = ['Missing' if v == 4 else v for v in listE]
    listE = ['Professional' if v == 5 else v for v in listE]
    listE = ['Sales' if v == 6 else v for v in listE]
    listE = ['Service' if v == 7 else v for v in listE]
    
    
    tupleE1 = entropyResult2.iloc[i,1]
    listE1 = list(tupleE1)
    listE1 = ['Agriculture' if v == 1 else v for v in listE1]
    listE1 = ['Crafts' if v == 2 else v for v in listE1]
    listE1 = ['Labor' if v == 3 else v for v in listE1]
    listE1 = ['Missing' if v == 4 else v for v in listE1]
    listE1 = ['Professional' if v == 5 else v for v in listE1]
    listE1 = ['Sales' if v == 6 else v for v in listE1]
    listE1 = ['Service' if v == 7 else v for v in listE1]

    combinationResult = '('
    for j in listE:
        combinationResult += j + ','
    combinationResult = combinationResult + ') , ('
    for j in listE1:
        combinationResult += j + ','
    combinationResult = combinationResult + ')'
    listCombination2.append(combinationResult)

listEntropy2 = list(entropyResult2.iloc[:,2])

entropyResult2 = DataFrame({'combination': listCombination2,
                    'entropy': listEntropy2})
print("|t 1f. The table:")
print(entropyResult2)

print("|t 1g. The optimal split for the JobCategory predictor: {}"
      .format(listCombination2[listEntropy2.index(min(listEntropy2))]))

print("|t 1h. The entropy() of the optimal split for the CreditCard predictor: {}"
      .format(min(listEntropy1)))

print("The entropy() of the optimal split for the JobCategory predictor: {}"
      .format(min(listEntropy2)))
