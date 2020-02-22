import numpy as np
import math
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
print("Mid-term")
print("|t Question1")
question1_data = [0.1811, 0.0775, 0.1279, 0.0045, 0.0001, 0.9457, 0.0021, 0,
                  0.0005, 0.7305, 0.8936]
q1, q2, q3, q4, q5 = np.percentile(question1_data, [0, 25, 50, 75, 100])
iqr = q4 - q2
print("The IQR of question1: {}".format(iqr))

print("|t Question3")
entropy = -((64/1000)*np.log2(64/1000) + (250/1000)*np.log2(250/1000) +
            (364/1000)*np.log2(364/1000) + (259/1000)*np.log2(259/1000) + (63/1000)*np.log2(63/1000))
print("The entropy of Question3: {}".format(round(entropy, 4)))

print("|t Question4")
Gini = 1 - ((0.2*0.2)*5)
print("The maximum Gini value: {}".format(round(Gini, 4)))


print("|t Question7")
import sklearn.metrics as metrics
Y = np.array(['Event',
                 'Non-Event',
                 'Non-Event',
                 'Event',
                 'Event',
                 'Non-Event',
                 'Event',
                 'Non-Event',
                 'Event',
                 'Event'])

nY = Y.shape[0]

predProbY = np.array([0.8, 0.5, 0.4, 0.6, 0.4, 0.7, 0.0, 0.5, 0.7, 0.6])
Y_true = 1.0 * np.isin(Y, ['Event'])
AUC = metrics.roc_auc_score(Y_true, predProbY)
print('Area Under Curve: {}' .format(round(AUC, 4)))

print("|t Question8")
# Determine the predicted class of Y
predY = np.empty_like(Y)
for i in range(nY):
    if (predProbY[i] >= 0.6):
        predY[i] = 'Event'
    else:
        predY[i] = 'Non-Event'
accuracy = metrics.accuracy_score(Y, predY)
print('Misclassification Rate: {}' .format(1-accuracy))

print("|t Question9")
from scipy.special import comb
print(comb(50, 4))

print("|t Question10") # 0.4944
import numpy as np
decileN = [421, 422, 422, 422, 421, 422, 422, 422, 422, 421]
gainN = [155, 52, 26, 19, 22, 27, 24, 19, 18, 22]
total = 0
decileNAcc = [421, 843, 1265, 1687, 2108, 2530, 2952, 3374, 3796, 4217]
gainNAcc = [155, 207, 233, 252, 274, 301, 325, 344, 362, 384]

AccRes = []

for i in range(10):
    AccRes.append(gainNAcc[i]/decileNAcc[i])
print(AccRes[3]/AccRes[9])

print("|t Question2")
no = [6815, 492, 212, 37]
yes = [2254, 312, 139, 41]


expectNo = []
expectYes = []
nIPlus = [9069, 804, 351, 78]

for i in range(4):
    expectNo.append(nIPlus[i]*(7556/10302))
    expectYes.append(nIPlus[i]*(2746/10302))
    
Result_1 = 0
Result_2 = 0

for i in range(4):
    Result_1 = Result_1 + (((no[i]-expectNo[i])**2)/expectNo[i])
    Result_2 = Result_2 + (((yes[i]-expectYes[i])**2)/expectYes[i])
Result = Result_1 + Result_2
    

observed = [no, yes]
expected = [expectNo, expectYes]
test = [[6815, 2254], [492, 312], [212, 139], [37, 41]]

from scipy import stats

stats.chi2_contingency(test)

print("|t Question3")
print(math.log(63/364))



