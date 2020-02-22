# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy
import statsmodels.api as stats
import pandas
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
import random


#Function getPredictedRing is used to calculate the predicted ring label 
def getPredictedRing(predProbList):
    maximum = 0
    index = -1
    for i in range(len(predProbList)):
        if(predProbList[i] > maximum):
            maximum = predProbList[i] 
            index = i
    return index

#Function ModelMetrics is used to calculate the misclassification rate and RASE
def ModelMetrics(
        predRing,
        observedRing,
        predProbRing):
    nObs = len(predRing)
    MisClassRate = 0
    for i in range(nObs):
        if (observedRing[i] != predRing[i]):
            MisClassRate = MisClassRate + 1
    MisClassRate /= nObs
    ASE_part = 0
    for i in range(nObs):
        for j in range(5):
            if (j == observedRing[i]):
                ASE_part = ASE_part + (1 - predProbRing[i ,j])**2
            else:
                ASE_part = ASE_part + (0 - predProbRing[i ,j])**2
        
    ASE = (1/(2*nObs))* ASE_part
    RASE = math.sqrt(ASE)   
    return(MisClassRate, RASE)
 
def drawScatter(predRingCategory):
    predRingDataFrame = pandas.DataFrame(data =predRingCategory, columns=['ring'] )
    predRing = x_logistic.join(predRingDataFrame)
    LABEL_COLOR_MAP = {0 : 'orange',
                       1 : 'green',
                       2 : 'blue',
                       3 : 'black',
                       4 : 'red'}
    label_color = [LABEL_COLOR_MAP[l] for l in predRingCategory]
    plt.figure(figsize = (16,10))
    plt.scatter(predRing[['x']], predRing[['y']], c = label_color)
    orange_patch= mpatches.Patch(color = 'orange', label ='0')
    green_patch= mpatches.Patch(color = 'green', label ='1')
    blue_patch= mpatches.Patch(color = 'blue', label ='2')
    black_patch= mpatches.Patch(color = 'black', label ='3')
    red_patch= mpatches.Patch(color = 'red', label ='4')
    plt.legend(handles=[orange_patch, green_patch, blue_patch, black_patch,red_patch ])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()
    
targetVar = ['ring']
fiveRingData = pandas.read_csv('FiveRing.csv', delimiter=',')
fiveRingData = fiveRingData.dropna()
y_logistic = fiveRingData['ring'].astype('category')
x_logistic = fiveRingData[['x','y']]
x_logistic = stats.add_constant(x_logistic, prepend=True)
#Build the multinomial logistic model
logit = stats.MNLogit(y_logistic, x_logistic)
trainLogisticFit = logit.fit(method='newton', full_output = True, maxiter = 1000, tol = 1e-8, disp=0)
#1a
thisParameter = trainLogisticFit.params
thisParameter = thisParameter.round(4)
print("|t 1a. The parameter estimates table: ", thisParameter)
logisticPredProb = trainLogisticFit.predict(x_logistic).values

logisticPredRing = []

for i in range(20010):
    index = -1
    index = getPredictedRing(logisticPredProb[i, 0:])
    logisticPredRing.append(index)

observedRing = list(fiveRingData['ring'])
#1b and 1c
MisClassRate, RASE = ModelMetrics(logisticPredRing, observedRing, logisticPredProb)
print("|t 1b. The Misclassification rate without bagging technque: {}".format(MisClassRate))
print("|t 1c. The Root Average Squared Error without bagging technque: {}".format(RASE))

#1d
print("|t 1d. Redraw the graph. The coloring scheme is 0 =orange, 1 = green, 2 = blue, 3 = black and 4 = red.")

drawScatter(logisticPredRing)


# Create a bootstrap sample
def sample_wr(inData):
    n = len(inData)
    outData = numpy.empty((n,1))
    for i in range(n):
        j = int(random.random() * n)
        outData[i] = inData[j]
    return outData

# Build the multinomial logistic model for the bootstrap samples
def bootstrap_MNlogit (x_train, y_train, x_test, nB):
   x_index = x_train.index
   nT = len(x_test)
   outProb = numpy.zeros((nT,5))
   # Initialize internal state of the random number generator
   random.seed(20190430)

   for iB in range(nB):
      bootIndex = sample_wr(x_index)
      x_train_boot = x_train.loc[bootIndex[:,0]]
      y_train_boot = y_train.loc[bootIndex[:,0]]
      logit = stats.MNLogit(y_train_boot, x_train_boot)
      trainLogisticFit = logit.fit(method='newton', full_output = True, maxiter = 1000, tol = 1e-8, disp=0)
      outProb = outProb + trainLogisticFit.predict(x_test)
   outProb = outProb / nB
   return outProb

#Execute the multinomial logistic model for the bootstrap samples
def execution(nB):
    logisticBootPredProb = bootstrap_MNlogit(x_logistic, y_logistic, x_logistic, nB).values
    logisticBootPredRing = []
    observedRing = list(fiveRingData['ring'])
    for i in range(20010):
        index = -1
        index = getPredictedRing(logisticBootPredProb[i, 0:])
        logisticBootPredRing.append(index)
    MisClassRate_nb, RASE_nb = ModelMetrics(logisticBootPredRing, observedRing, logisticBootPredProb)
    return(MisClassRate_nb, RASE_nb, logisticBootPredRing)

(MisClassRate_nb10, RASE_nb10, logisticBootPredRing_nb10) = execution(10)
(MisClassRate_nb20, RASE_nb20, logisticBootPredRing_nb20) = execution(20)
(MisClassRate_nb30, RASE_nb30, logisticBootPredRing_nb30) = execution(30)
(MisClassRate_nb40, RASE_nb40, logisticBootPredRing_nb40) = execution(40)
(MisClassRate_nb50, RASE_nb50, logisticBootPredRing_nb50) = execution(50)
(MisClassRate_nb60, RASE_nb60, logisticBootPredRing_nb60) = execution(60)
(MisClassRate_nb70, RASE_nb70, logisticBootPredRing_nb70) = execution(70)
(MisClassRate_nb80, RASE_nb80, logisticBootPredRing_nb80) = execution(80)
(MisClassRate_nb90, RASE_nb90, logisticBootPredRing_nb90) = execution(90)
(MisClassRate_nb100, RASE_nb100, logisticBootPredRing_nb100) = execution(100)
# 1e
tableMAndRASENB = pandas.DataFrame({
                                        'Bootstraps':
                                            ['0','10','20','30','40','50','60','70','80','90','100'],
        
                                        'MisClassRate': [MisClassRate,
                                           MisClassRate_nb10,
                                           MisClassRate_nb20,
                                           MisClassRate_nb30,
                                           MisClassRate_nb40,
                                           MisClassRate_nb50,
                                           MisClassRate_nb60,
                                           MisClassRate_nb70,
                                           MisClassRate_nb80,
                                           MisClassRate_nb90,
                                           MisClassRate_nb100],
                            'RASE': [RASE,
                                     RASE_nb10,
                                     RASE_nb20,
                                     RASE_nb30,
                                     RASE_nb40,
                                     RASE_nb50,
                                     RASE_nb60,
                                     RASE_nb70,
                                     RASE_nb80,
                                     RASE_nb90,
                                     RASE_nb100]
                          })

tableMAndRASENB.set_index('Bootstraps', inplace = True) 
        
print("|t 1e. List the Misclassification Rate and the Root Average Squared Error of the bootstrap results, include the no-bootstrap metrics:",tableMAndRASENB)

#1f
print("|t 1f. Redraw the graph. The coloring scheme is 0 =orange, 1 = green, 2 = blue, 3 = black and 4 = red.")
print("Number of bootstraps equals to 10: ")
drawScatter(logisticBootPredRing_nb10)
print("Number of bootstraps equals to 20: ")
drawScatter(logisticBootPredRing_nb20)
print("Number of bootstraps equals to 30: ")
drawScatter(logisticBootPredRing_nb30)
print("Number of bootstraps equals to 40: ")
drawScatter(logisticBootPredRing_nb40)
print("Number of bootstraps equals to 50: ")
drawScatter(logisticBootPredRing_nb50)
print("Number of bootstraps equals to 60: ")
drawScatter(logisticBootPredRing_nb60)
print("Number of bootstraps equals to 70: ")
drawScatter(logisticBootPredRing_nb70)
print("Number of bootstraps equals to 80: ")
drawScatter(logisticBootPredRing_nb80)
print("Number of bootstraps equals to 90: ")
drawScatter(logisticBootPredRing_nb90)
print("Number of bootstraps equals to 100: ")
drawScatter(logisticBootPredRing_nb100)
