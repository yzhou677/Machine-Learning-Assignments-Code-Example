#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 14:47:10 2019

@author: yuqi
"""

import matplotlib.pyplot as plt
import numpy
import pandas
import sklearn.model_selection as model_selection
import statsmodels.api as stats
import sklearn.metrics as metrics

targetVar = ['CLAIM_FLAG']
nominalVar = ['CREDIT_SCORE_BAND']
intervalVar = ['BLUEBOOK_1000', 'CUST_LOYALTY', 'MVR_PTS', 'TIF', 'TRAVTIME']

policyData = pandas.read_csv('policy_2001.csv',
                            delimiter=',',
                            usecols = (targetVar + nominalVar + intervalVar))
inputData = policyData.dropna()
yData = inputData[targetVar].astype('category')
xData = inputData[nominalVar].astype('category')
xData = pandas.get_dummies(xData)
xData = xData.join(inputData[intervalVar]) 
x_train, x_test, y_train, y_test = model_selection.train_test_split(xData, yData, test_size = 0.25, random_state = 20190402,stratify = yData)


from sklearn import tree
classTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=20190402)
policyDataTrain_DT = classTree.fit(x_train, y_train)


y_pred_decision_tree = classTree.predict_proba(x_test)
y_pred_decision_tree = pandas.DataFrame(y_pred_decision_tree)


x_train = stats.add_constant(x_train, prepend=True)

x_test = stats.add_constant(x_test, prepend=True)


logit = stats.MNLogit(y_train, x_train)
thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
y_pred_logit = thisFit.predict(x_test)

Y = []
y_test = numpy.array(y_test) 
for i in range(155):
    if(y_test[i] == 1):
        Y.append('Event')
    else:
        Y.append('Non-Event')
        
Y =numpy.array(Y)
nY = Y.shape[0]

predProbY = y_pred_logit[1].values
predProbY = numpy.array(predProbY)

# Determine the predicted class of Y
predY = numpy.empty_like(Y)
for i in range(nY):
    if (predProbY[i]>= 0.287879):
        predY[i] = 'Event'
    else:
        predY[i] = 'Non-Event'

# Calculate the Root Average Squared Error
RASE = 0.0
for i in range(nY):
    if (Y[i] == 'Event'):
        RASE += (1 - predProbY[i])**2
    else:
        RASE += (0 - predProbY[i])**2
RASE = numpy.sqrt(RASE/nY)

# Calculate the Root Mean Squared Error
Y_true = 1.0 * numpy.isin(Y, ['Event'])
RMSE = metrics.mean_squared_error(Y_true, predProbY)
RMSE = numpy.sqrt(RMSE)

# For binary y_true, y_score is supposed to be the score of the class with greater label.
AUC = metrics.roc_auc_score(Y_true, predProbY)
accuracy = metrics.accuracy_score(Y, predY)

print('                  Accuracy: {:.13f}' .format(accuracy))
print('    Misclassification Rate: {:.13f}' .format(1-accuracy))
print('          Area Under Curve: {:.13f}' .format(AUC))
print('Root Average Squared Error: {:.13f}' .format(RASE))
print('   Root Mean Squared Error: {:.13f}' .format(RMSE))

    
# Generate the coordinates for the ROC curve
OneMinusSpecificity, Sensitivity, thresholds = metrics.roc_curve(Y, predProbY, pos_label = 'Event')

# Add two dummy coordinates
OneMinusSpecificity = numpy.append([0], OneMinusSpecificity)
Sensitivity = numpy.append([0], Sensitivity)

OneMinusSpecificity = numpy.append(OneMinusSpecificity, [1])
Sensitivity = numpy.append(Sensitivity, [1])



predProbY_tree = y_pred_decision_tree[1].values
predProbY_tree = numpy.array(predProbY_tree)
predY_tree = numpy.empty_like(Y)

for i in range(nY):
    if (predProbY_tree[i]>= 0.287879):
        predY_tree[i] = 'Event'
    else:
        predY_tree[i] = 'Non-Event'
        
# Calculate the Root Average Squared Error
RASE_tree = 0.0
for i in range(nY):
    if (Y[i] == 'Event'):
        RASE_tree += (1 - predProbY_tree[i])**2
    else:
        RASE_tree += (0 - predProbY_tree[i])**2
RASE_tree = numpy.sqrt(RASE_tree/nY)

# Calculate the Root Mean Squared Error
Y_true_tree = 1.0 * numpy.isin(Y, ['Event'])
RMSE_tree = metrics.mean_squared_error(Y_true_tree, predProbY_tree)
RMSE_tree = numpy.sqrt(RMSE_tree)

AUC_tree = metrics.roc_auc_score(Y_true_tree, predProbY_tree)
accuracy_tree = metrics.accuracy_score(Y, predY_tree)
print('                  Accuracy: {:.13f}' .format(accuracy_tree))
print('    Misclassification Rate: {:.13f}' .format(1-accuracy_tree))
print('          Area Under Curve: {:.13f}' .format(AUC_tree))
print('Root Average Squared Error: {:.13f}' .format(RASE_tree))
print('   Root Mean Squared Error: {:.13f}' .format(RMSE_tree))

# Generate the coordinates for the ROC curve
OneMinusSpecificity_tree, Sensitivity_tree, thresholds_tree = metrics.roc_curve(Y, predProbY_tree, pos_label = 'Event')

# Add two dummy coordinates
OneMinusSpecificity_tree = numpy.append([0], OneMinusSpecificity_tree)
Sensitivity_tree = numpy.append([0], Sensitivity_tree)

OneMinusSpecificity_tree = numpy.append(OneMinusSpecificity_tree, [1])
Sensitivity_tree = numpy.append(Sensitivity_tree, [1])

'''
# Draw the ROC curve
plt.figure(figsize=(6,6))
plt.plot(OneMinusSpecificity, Sensitivity, marker = 'o',label = 'Logistic',
         color = 'blue', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.plot(OneMinusSpecificity_tree, Sensitivity_tree, marker = 'o',
         label = 'Decision tree',color = 'green', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.plot([0, 1], [0, 1], color = 'red', linestyle = ':')
plt.title("Receiver Operating Characteristic Curve")
plt.grid(True)
plt.xlabel("1 - Specificity (False Positive Rate)")
plt.ylabel("Sensitivity (True Positive Rate)")
ax = plt.gca()
ax.set_aspect('equal')
plt.savefig("11-d_ds")
plt.show()
'''

plt.figure(figsize=(6,6))
plt.plot(OneMinusSpecificity_tree, Sensitivity_tree, marker = 'o', label = 'Decision Tree',
         color = 'green', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.plot(OneMinusSpecificity, Sensitivity, marker = 'o', label = 'Logistic',
         color = 'blue', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.plot([0, 1], [0, 1], color = 'red', linestyle = ':')
plt.title("Receiver Operating Characteristic Curve")
plt.grid(True)
plt.xlabel("1 - Specificity (False Positive Rate)")
plt.ylabel("Sensitivity (True Positive Rate)")
plt.axis(aspect = 'equal')
plt.legend()
plt.savefig("11-d_ds")
print("figure saved")


