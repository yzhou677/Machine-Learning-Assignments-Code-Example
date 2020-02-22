#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 14:00:30 2019

@author: yuqi
"""
import numpy
import sklearn.tree as tree
import pandas
import matplotlib.pyplot as plt
import math
import matplotlib.patches as mpatches

#Function getPredictedRing is used to calculate the predicted ring label 
def getPredictedRing(predProbList):
    index = numpy.argmax(numpy.array(predProbList))
    return index

#Function ModelMetrics is used to calculate the misclassification rate and RASE
def ModelMetrics(
        treePredRing,
        observedRing,
        treePredProbRing):
    nObs = len(treePredRing)
    MisClassRate = 0
    for i in range(nObs):
        if (observedRing[i] != treePredRing[i]):
            MisClassRate = MisClassRate + 1
    MisClassRate /= nObs
    ASE_part = 0
    for i in range(nObs):
        for j in range(5):
            if (j == observedRing[i]):
                ASE_part = ASE_part + (1 - treePredProbRing[i ,j])**2
            else:
                ASE_part = ASE_part + (0 - treePredProbRing[i ,j])**2
        
    ASE = (1/(2*nObs))* ASE_part
    RASE = math.sqrt(ASE)   
    return(MisClassRate, RASE)
    
def drawScatter(predRingCategory):
    predRingDataFrame = pandas.DataFrame(data =predRingCategory, columns=['ring'] )
    predRing = x_classTrain.join(predRingDataFrame)
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
y_classTrain = fiveRingData['ring'].astype('category')
x_classTrain = fiveRingData[['x','y']]
y_classTrain = list(y_classTrain)
# Build the classification tree model 
classTree = tree.DecisionTreeClassifier(criterion = 'entropy',
                                        max_depth = 2,
                                        random_state = 20190415)
trainTreeFit = classTree.fit(x_classTrain, y_classTrain)
treePredProbRing = classTree.predict_proba(x_classTrain)

treePredRing = []
# Get the predicted ring labels
for i in range(20010):
    index = -1
    index = getPredictedRing(treePredProbRing[i, 0:])
    treePredRing.append(index)
    
observedRing = list(fiveRingData['ring'])
#2a 2b
(MisClassRate, RASE) = ModelMetrics(treePredRing, observedRing, treePredProbRing)
print("|t 2a. The Misclassification rate without boosting: {}".format(MisClassRate))
print("|t 2b. The Root Average Squared Error without boosting: {}".format(RASE))

#2c
print("|t 2c. Redraw the graph. The coloring scheme is 0 =orange, 1 = green, 2 = blue, 3 = black and 4 = red.")
drawScatter(treePredRing)

def iterationBoosting(iteration):
    predProbList = []
    accuracyList = []
    w_train = numpy.ones((20010), dtype = float)
    for iter in range(iteration):
        classTree = tree.DecisionTreeClassifier(criterion = 'entropy',
                                                max_depth = 2,
                                                random_state = 20190415)
        treeFit = classTree.fit(x_classTrain, y_classTrain, w_train)
        treePredProb = classTree.predict_proba(x_classTrain)
        accuracy = classTree.score(x_classTrain, y_classTrain, w_train)
        accuracyList.append(accuracy) 
        predProbList.append(treePredProb)
        
        treePredRing = []
        for i in range(20010):
            index = -1
            index = getPredictedRing(treePredProb[i, 0:])
            treePredRing.append(index)
        #Calculate the misclassification rate and RASE
        (MisClassRate_tree_w_1, RASE_tree_w_1) = ModelMetrics(treePredRing, observedRing, treePredProb)
        #If MisClassRate is 0 then the iteration stops
        if (MisClassRate_tree_w_1 == 0):
            break;   
        # Update the weights
        for i in range(20010):
            if (treePredRing[i] == observedRing[i]):
                weight_part = 0
                for j in range(5):
                    if (j == observedRing[i]):
                        weight_part += abs(1 - treePredProb[i, j])
                    else:
                        weight_part += abs(0 - treePredProb[i, j])
                w_train[i] = (1/5)*weight_part
            else:
                weight_part = 0
                for j in range(5):
                    if (j == observedRing[i]):
                        weight_part += abs(1 - treePredProb[i, j])
                    else:
                        weight_part += abs(0 - treePredProb[i, j])
                w_train[i] = 1 + (1/5)*weight_part
    totalAccuracy = 0
    for i in range(len(accuracyList)):
        totalAccuracy = totalAccuracy + accuracyList[i]
    finalPredProb = accuracyList[0] * predProbList[0]
    for j in range(1, len(accuracyList)):
        finalPredProb = finalPredProb + accuracyList[j] * predProbList[j]
    finalPredProb = finalPredProb/totalAccuracy
    finalTreePredRing = []
    for i in range(20010):
        index = -1
        index = getPredictedRing(finalPredProb[i, 0:])
        finalTreePredRing.append(index)
    #Calculate the misclassification rate and RASE
    (MisClassRate_tree_w, RASE_tree_w) = ModelMetrics(finalTreePredRing, observedRing, finalPredProb)
    
    return(MisClassRate_tree_w, RASE_tree_w, finalTreePredRing)
                
(MisClassRate_tree_w100, RASE_tree_w100, treePredRing_tree100) = iterationBoosting(100)
drawScatter(treePredRing_tree100)

(MisClassRate_tree_w200, RASE_tree_w200, treePredRing_tree200) = iterationBoosting(200)
drawScatter(treePredRing_tree200)
(MisClassRate_tree_w300, RASE_tree_w300, treePredRing_tree300) = iterationBoosting(300)
drawScatter(treePredRing_tree300)

(MisClassRate_tree_w400, RASE_tree_w400, treePredRing_tree400) = iterationBoosting(400)
drawScatter(treePredRing_tree400)

(MisClassRate_tree_w500, RASE_tree_w500, treePredRing_tree500) = iterationBoosting(500)
drawScatter(treePredRing_tree500)

(MisClassRate_tree_w600, RASE_tree_w600, treePredRing_tree600) = iterationBoosting(600)
drawScatter(treePredRing_tree600)

(MisClassRate_tree_w700, RASE_tree_w700, treePredRing_tree700) = iterationBoosting(700)
drawScatter(treePredRing_tree700)

(MisClassRate_tree_w800, RASE_tree_w800, treePredRing_tree800) = iterationBoosting(800)
drawScatter(treePredRing_tree800)

(MisClassRate_tree_w900, RASE_tree_w900, treePredRing_tree900) = iterationBoosting(900)
drawScatter(treePredRing_tree900)

(MisClassRate_tree_w1000, RASE_tree_w1000, treePredRing_tree1000) = iterationBoosting(1000)
drawScatter(treePredRing_tree1000)


tableMAndRASEIT = pandas.DataFrame({'MisClassRate': [MisClassRate,
                                           MisClassRate_tree_w100,
                                           MisClassRate_tree_w200,
                                           MisClassRate_tree_w300,
                                           MisClassRate_tree_w400,
                                           MisClassRate_tree_w500,
                                           MisClassRate_tree_w600,
                                           MisClassRate_tree_w700,
                                           MisClassRate_tree_w800,
                                           MisClassRate_tree_w900,
                                           MisClassRate_tree_w1000],
                            'RASE': [RASE,
                                     RASE_tree_w100,
                                     RASE_tree_w200,
                                     RASE_tree_w300,
                                     RASE_tree_w400,
                                     RASE_tree_w500,
                                     RASE_tree_w600,
                                     RASE_tree_w700,
                                     RASE_tree_w800,
                                     RASE_tree_w900,
                                     RASE_tree_w1000],
                            'Iterations': ['0','100','200','300','400','500','600','700','800','900','1000']
                          })

tableMAndRASEIT.set_index('Iterations', inplace = True) 
print("|t 2d. List the Misclassification Rate and the Root Average Squared Error of the boosting results:")
print(tableMAndRASEIT)
print("|t 2e. 100 iterations: ")
drawScatter(treePredRing_tree100)
print("|t 2e. 200 iterations: ")
drawScatter(treePredRing_tree200)
print("|t 2e. 300 iterations: ")
drawScatter(treePredRing_tree300)
print("|t 2e. 400 iterations: ")
drawScatter(treePredRing_tree400)
print("|t 2e. 500 iterations: ")
drawScatter(treePredRing_tree500)
print("|t 2e. 600 iterations: ")
drawScatter(treePredRing_tree600)
print("|t 2e. 700 iterations: ")
drawScatter(treePredRing_tree700)
print("|t 2e. 800 iterations: ")
drawScatter(treePredRing_tree800)
print("|t 2e. 900 iterations: ")
drawScatter(treePredRing_tree900)
print("|t 2e. 1000 iterations: ")
drawScatter(treePredRing_tree1000)



