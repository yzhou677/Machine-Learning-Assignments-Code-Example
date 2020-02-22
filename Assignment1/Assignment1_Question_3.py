import numpy as np
from numpy import linalg as LA
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors as kNN
from sklearn import neighbors
from pandas import DataFrame

fraudData = pd.read_csv('Fraud.csv')
#print(fraudData)

fraud_field = fraudData.FRAUD.values
target = fraudData['FRAUD']
#print(fraud_field)

fraudulent = 0 
for fraud in fraud_field:
    fraudulent = fraudulent + fraud
percent_fraudulent = fraudulent/len(fraud_field)

print("|t 3a. The percent of investigations are found to be fraudulent: {}".format(round(percent_fraudulent, 4)))

def draw_boxplot(column_name):
    fraudData.boxplot(column=column_name, by='FRAUD', vert=False)
    plt.suptitle("")
    plt.xlabel(column_name)
    plt.ylabel("FRAUD")
    plt.grid(axis="y")
    plt.show()

#The graphs of Question 3b 
draw_boxplot('TOTAL_SPEND')
draw_boxplot('DOCTOR_VISITS')
draw_boxplot('NUM_CLAIMS')
draw_boxplot('MEMBER_DURATION')
draw_boxplot('OPTOM_PRESC')
draw_boxplot('NUM_MEMBERS')

trainData = fraudData[['TOTAL_SPEND','DOCTOR_VISITS','NUM_CLAIMS','MEMBER_DURATION','OPTOM_PRESC','NUM_MEMBERS']]

fraud_matrix =  np.matrix(trainData.values)
#print(fraud_matrix[1,:])

#print("Number of Dimensions = ", fraud_matrix.ndim)
fraud_matrixtfraud_matrix = fraud_matrix.transpose()*fraud_matrix

#Eigenvalue decomposition
evals, evecs = np.linalg.eigh(fraud_matrixtfraud_matrix)

#The answer of Question 3c
print("|t 3c i. How many dimensions are used: {}".format(6))
print(evals)

#Here is the transformation matrix
transf = evecs*np.linalg.inv(np.sqrt(np.diagflat(evals)))
print("3c ii. Transformation Matrix = \n", transf)

#Here is the transformed fraud_matrix
fraud_matrix_transf = fraud_matrix * transf
#print("The Transformed fraud_matrix=\n", fraud_matrix_transf)

#Check columns of transformed fraud_matrix
fraud_matrixtfraud_matrix = fraud_matrix_transf.transpose()*fraud_matrix_transf
print("Expect an Identity Matrix=\n", fraud_matrixtfraud_matrix)

#Specify the kNN 
kNNSpec = neighbors.KNeighborsClassifier(n_neighbors = 5, algorithm = 'brute', metric = 'euclidean')

#Build nearest neighbors
nbrs = kNNSpec.fit(fraud_matrix_transf, target)
# fraud_matrix_transf

score = nbrs.score(fraud_matrix_transf, target)

#The answer of Question 3d
print("|t 3d. The score function return value: {}".format(score))

distance, indices = nbrs.kneighbors(fraud_matrix_transf)

fraud_observation = [[7500, 15, 3, 127, 2, 2]]
#print(observation_matrix)
observation_matrix_transf = fraud_observation * transf
myNeighbors = nbrs.kneighbors(observation_matrix_transf, return_distance = False)

#The answer of Question 3e
print("|t 3e. My neighbors = \n", myNeighbors)

print(trainData.iloc[588])
print(trainData.iloc[2897])
print(trainData.iloc[1199])
print(trainData.iloc[1246])
print(trainData.iloc[886])
#print(fraudData.loc[886])

'''
class_prob = nbrs.predict_proba(fraud_matrix_transf)
print(class_prob)



fraud_group = fraudData.groupby('FRAUD')
total_list_0 = sorted(fraud_group.get_group(0).TOTAL_SPEND.values)
total_list_1 = sorted(fraud_group.get_group(1).TOTAL_SPEND.values)

df1 = DataFrame({
'0' : total_list_0
})

df2 = DataFrame({'1': total_list_1})
frames = [df1, df2]
result = pd.concat(frames, sort=False)
#print(result)

result.boxplot(column=['0', '1'], vert=False)
plt.suptitle("")
plt.xlabel("TOTAL_SPEND")
plt.ylabel("FRAUD")
plt.grid(axis="y")
plt.show()

'''


'''
#Misclassification Rate
targetClass = [0, 1]

nMissClass = 0
for i in range(trainData.shape[0]):
    j = np.argmax(class_prob[i][:])
    predictClass = targetClass[j]
    if(predictClass != target.iloc[i]):
        nMissClass += 1

print(nMissClass)

rateMissClass = nMissClass / trainData.shape[0]
print('Misclassification Rate = ', rateMissClass)
'''
