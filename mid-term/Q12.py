
import numpy
import sklearn.metrics as metrics
import sklearn.model_selection as model_selection
import sklearn.neighbors as neighbors
import sklearn.tree as tree
import statsmodels.api as stats
import pandas
import matplotlib.pyplot as plt
#=========================/data_preprocess===================================
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

#=========================/Procedures===================================
print("<12-a>")
print("=========================================")
print('Training Observations:', x_train.shape[0])
print('Testing Ovservations:', x_test.shape[0])

print("<12-b>")
print("=========================================")
rateCLAIM = len(y_train[y_train['CLAIM_FLAG'] == 1]) / len(y_train)
print('Result', rateCLAIM)

print("<12-c>")
print("=========================================")
classTree = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 5, random_state = 20190402)
trainTreeFit = classTree.fit(x_train, y_train)

testTreePredProb = classTree.predict_proba(x_test)
DepVar = y_test['CLAIM_FLAG'].values
EventValue = 1
EventPredProb = testTreePredProb[:,1]
Threshold = rateCLAIM
AUC = metrics.roc_auc_score(DepVar, EventPredProb)
nObs = len(DepVar)
RASE = 0
MisClassRate = 0
for i in range(nObs):
    p = EventPredProb[i]
    if (DepVar[i] == EventValue):
        RASE += (1.0 - p)**2
        if (p < Threshold):
            MisClassRate += 1
    else:
        RASE += p**2
        if (p >= Threshold):
            MisClassRate += 1
RASE = numpy.sqrt(RASE / nObs)
MisClassRate /= nObs
testTreeAUC =AUC
testTreeRASE =RASE
testTreeMisClassRate = MisClassRate

testTreeFP, testTreeTP, testTreeThresholds = metrics.roc_curve(y_test['CLAIM_FLAG'], testTreePredProb[:,1], pos_label = 1)
#testTreeLift, testTreeAccLift = compute_lift_coordinates 
DepVar= y_test['CLAIM_FLAG'].values
EventValue=1
EventPredProb = testTreePredProb[:,1]
nObs = len(DepVar)
quantileCutOff = numpy.percentile(EventPredProb, numpy.arange(0, 100, 10))
nQuantile = len(quantileCutOff)
quantileIndex = numpy.zeros(nObs)
for i in range(nObs):
    iQ = nQuantile
    EPP = EventPredProb[i]
    for j in range(1, nQuantile):
        if (EPP > quantileCutOff[-j]):
            iQ -= 1
    quantileIndex[i] = iQ
countTable = pandas.crosstab(quantileIndex, DepVar)
decileN = countTable.sum(1)
decilePct = 100 * (decileN / nObs)
gainN = countTable[EventValue]
totalNResponse = gainN.sum(0)
gainPct = 100 * (gainN /totalNResponse)
responsePct = 100 * (gainN / decileN)
overallResponsePct = 100 * (totalNResponse / nObs)
lift = responsePct / overallResponsePct
LiftCoordinates = pandas.concat([decileN, decilePct, gainN, gainPct, responsePct, lift],
                                axis = 1, ignore_index = True)
LiftCoordinates = LiftCoordinates.rename({0:'Decile N',
                                          1:'Decile %',
                                          2:'Gain N',
                                          3:'Gain %',
                                          4:'Response %',
                                          5:'Lift'}, axis = 'columns')
accCountTable = countTable.cumsum(axis = 0)
decileN = accCountTable.sum(1)
decilePct = 100 * (decileN / nObs)
gainN = accCountTable[EventValue]
gainPct = 100 * (gainN / totalNResponse)
responsePct = 100 * (gainN / decileN)
lift = responsePct / overallResponsePct

accLiftCoordinates = pandas.concat([decileN, decilePct, gainN, gainPct, responsePct, lift],
                                   axis = 1, ignore_index = True)
accLiftCoordinates = accLiftCoordinates.rename({0:'Acc. Decile N',
                                                1:'Acc. Decile %',
                                                2:'Acc. Gain N',
                                                3:'Acc. Gain %',
                                                4:'Acc. Response %',
                                                5:'Acc. Lift'}, axis = 'columns')
testTreeLift = LiftCoordinates
testTreeAccLift = accLiftCoordinates
x_trainP1 = stats.add_constant(x_train, prepend=True)
print()
print('From Tree model')
print("------------------------------------------------------")
print('          Area Under Curve = {:.6f}' .format(testTreeAUC))
print('Root Average Squared Error = {:.6f}' .format(testTreeRASE))
print('    Misclassification Rate = {:.6f}' .format(testTreeMisClassRate))
print("-------------------------------------------------------")



logit = stats.MNLogit(y_train, x_trainP1)
trainLogisticFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)

x_testP1 = stats.add_constant(x_test, prepend=True)
testLogisticPredProb = trainLogisticFit.predict(x_testP1).values
DepVar = y_test['CLAIM_FLAG'].values
EventValue = 1
EventPredProb = testLogisticPredProb[:,1]
Threshold = rateCLAIM
AUC = metrics.roc_auc_score(DepVar, EventPredProb)
nObs = len(DepVar)
RASE = 0
MisClassRate = 0
for i in range(nObs):
    p = EventPredProb[i]
    if (DepVar[i] == EventValue):
        RASE += (1.0 - p)**2
        if (p < Threshold):
            MisClassRate += 1
    else:
        RASE += p**2
        if (p >= Threshold):
            MisClassRate += 1
RASE = numpy.sqrt(RASE / nObs)
MisClassRate /= nObs
testLogisticAUC =AUC
testLogisticRASE =RASE
testLogisticMisClassRate = MisClassRate

testLogisticFP, testLogisticTP, testLogisticThresholds = metrics.roc_curve(y_test['CLAIM_FLAG'], testLogisticPredProb[:,1], pos_label = 1)

DepVar= y_test['CLAIM_FLAG'].values
EventValue=1
EventPredProb = testTreePredProb[:,1]
nObs = len(DepVar)
quantileCutOff = numpy.percentile(EventPredProb, numpy.arange(0, 100, 10))
nQuantile = len(quantileCutOff)
quantileIndex = numpy.zeros(nObs)
for i in range(nObs):
    iQ = nQuantile
    EPP = EventPredProb[i]
    for j in range(1, nQuantile):
        if (EPP > quantileCutOff[-j]):
            iQ -= 1
    quantileIndex[i] = iQ
countTable = pandas.crosstab(quantileIndex, DepVar)
decileN = countTable.sum(1)
decilePct = 100 * (decileN / nObs)
gainN = countTable[EventValue]
totalNResponse = gainN.sum(0)
gainPct = 100 * (gainN /totalNResponse)
responsePct = 100 * (gainN / decileN)
overallResponsePct = 100 * (totalNResponse / nObs)
lift = responsePct / overallResponsePct
LiftCoordinates = pandas.concat([decileN, decilePct, gainN, gainPct, responsePct, lift],
                                axis = 1, ignore_index = True)
LiftCoordinates = LiftCoordinates.rename({0:'Decile N',
                                          1:'Decile %',
                                          2:'Gain N',
                                          3:'Gain %',
                                          4:'Response %',
                                          5:'Lift'}, axis = 'columns')
accCountTable = countTable.cumsum(axis = 0)
decileN = accCountTable.sum(1)
decilePct = 100 * (decileN / nObs)
gainN = accCountTable[EventValue]
gainPct = 100 * (gainN / totalNResponse)
responsePct = 100 * (gainN / decileN)
lift = responsePct / overallResponsePct

accLiftCoordinates = pandas.concat([decileN, decilePct, gainN, gainPct, responsePct, lift],
                                   axis = 1, ignore_index = True)
accLiftCoordinates = accLiftCoordinates.rename({0:'Acc. Decile N',
                                                1:'Acc. Decile %',
                                                2:'Acc. Gain N',
                                                3:'Acc. Gain %',
                                                4:'Acc. Response %',
                                                5:'Acc. Lift'}, axis = 'columns')
testLogisticLift = LiftCoordinates
testLogisticAccLift = accLiftCoordinates

x_trainP1 = stats.add_constant(x_train, prepend=True)

print('From Logistic model')
print("------------------------------------------------------")
print('          Area Under Curve = {:.6f}' .format(testLogisticAUC))
print('Root Average Squared Error = {:.6f}' .format(testLogisticRASE))
print('    Misclassification Rate = {:.6f}'.format(testLogisticMisClassRate))
print("-------------------------------------------------------")

print("<12-d>")
print("=========================================")
plt.figure(figsize=(6,6))
plt.plot(testTreeFP, testTreeTP, marker = 'o', label = 'Decision Tree',
         color = 'blue', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.plot(testLogisticFP, testLogisticTP, marker = 'o', label = 'Logistic',
         color = 'green', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.plot([0, 1], [0, 1], color = 'red', linestyle = ':')
plt.title("Receiver Operating Characteristic Curve")
plt.grid(True)
plt.xlabel("1 - Specificity (False Positive Rate)")
plt.ylabel("Sensitivity (True Positive Rate)")
plt.axis(aspect = 'equal')
plt.legend()
plt.savefig("11-d_ds")
print("figure saved")

