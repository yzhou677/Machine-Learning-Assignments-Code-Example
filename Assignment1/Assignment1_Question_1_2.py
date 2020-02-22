import pandas
import matplotlib.pyplot as plt
import numpy as np
import math
from pandas import DataFrame

#Read in data
trainData = pandas.read_csv('NormalSample.csv')

#Remove all missing observations
# trainData = trainData.dropna()
#print(trainData)

x_field = trainData.x.values
#print(x_field)
min_x = min(x_field)
max_x = max(x_field)
x_range_value = max_x - min_x
x_sorted = sorted(x_field)
#print(x_sorted)

q1, q2, q3, q4, q5 = np.percentile(x_field, [0, 25, 50, 75, 100])
#print(q1, q2, q3, q4, q5)

iqr = q4 - q2
#print("iqr", iqr)

N_pow = math.pow(1001, -1/3)
#print(N_pow)
h = 2 * iqr * N_pow

#The answer of Question 1a 
print("|t 1a. Recommended bin-width: {}".format(h))

#The answer of Question 1b
print("|t 1b. The minimum and maximum values of the field x: {} {}".format(min_x, max_x)) 

a = int((min_x*10)/10)
b = int(((max_x + 1)*10)/10)

#The answer of Question 1c
print("|t 1c. The values of a and b: {} {}".format(a, b))

whisker_1 = q2 - 1.5 * iqr
if whisker_1 < q1 :
    whisker_1 = q1
whisker_2 = q4 + 1.5 * iqr
if whisker_2 > q5:
    whisker_2 = q5

#The answer of Question 2a
print("|t 2a. The five-number summary of x: {} {} {} {} {}".format(q1, q2, q3, q4, q5))
print("The values of the 1.5 IQR whiskers: {} {}".format(whisker_1, whisker_2))

trainData_by_group = trainData.groupby('group')
x_list_group0 = sorted(trainData_by_group.get_group(0).x.values)
#print(x_list_group0)

x_list_group1 = sorted(trainData_by_group.get_group(1).x.values)
#print(x_list_group1)

q100_group0, q75_group0, q50_group0, q25_group0, q0_group0 = np.percentile(x_list_group0, [100, 75, 50, 25, 0])
iqr_group0 = q75_group0 - q25_group0

whisker_1_group0 = q25_group0 - 1.5 * iqr_group0
if whisker_1_group0 < q0_group0:
    whisker_1_group0 = q0_group0

whisker_2_group0 = q75_group0 + 1.5 * iqr_group0
if whisker_2_group0 > q100_group0:
    whisker_2_group0 = q100_group0

#The answer of Question 2b(group0)
print("|t 2b. The five-number summary of x in group0: {} {} {} {} {}".format(q0_group0 ,q25_group0 ,q50_group0 ,q75_group0 ,q100_group0))
print("The values of the 1.5 IQR whiskers of group0: {} {}".format(whisker_1_group0, whisker_2_group0))

q100_group1, q75_group1, q50_group1, q25_group1,q0_group1 = np.percentile(x_list_group1, [100, 75, 50, 25, 0])
iqr_group1 = q75_group1 - q25_group1

whisker_1_group1 = q25_group1 - 1.5 * iqr_group1
if whisker_1_group1 < q0_group1:
    whisker_1_group1 = q0_group1

whisker_2_group1 = q75_group1 + 1.5 * iqr_group1
if whisker_2_group1 > q100_group1:
    whisker_2_group1 = q100_group1

#The answer of Question 2b(group1)
print("The five-number summary of x in group1: {} {} {} {} {}".format(q0_group1, q25_group1, q50_group1, q75_group1, q100_group1))
print("The values of the 1.5 IQR whiskers of group1: {} {}".format(whisker_1_group1, whisker_2_group1))

#The graph of Question 2c
#Visualize the boxplot of the field x
trainData.boxplot(column='x', vert=False)
plt.title("Boxplot of the x field")
plt.suptitle("")
plt.xlabel("x")
plt.grid(axis="y")
plt.show()


df1 = DataFrame({
'x': x_field
})
df2 = DataFrame({'x_group0': x_list_group0})
df3 = DataFrame({'x_group1': x_list_group1})
frames = [df1, df2, df3]
result = pandas.concat(frames, sort=False)

#print(result)

#The graph of Question 2d
result.boxplot(column=['x', 'x_group0', 'x_group1'], vert=False)
plt.title("Boxplot of the x field")
plt.suptitle("")
plt.xlabel("x")
plt.grid(axis="y")
plt.show()



def outliers_identify_index(x, lower_bound, upper_bound):
    return np.where((x > upper_bound) | (x < lower_bound))

def outliers_identify(x, lower_bound, upper_bound):
    outliers_index = outliers_identify_index(x, lower_bound, upper_bound)[0]
    outliers = []
    for i in outliers_index:
        outliers.append(x[i])
    return outliers

#The answer of Question 2d(identify outliers)
outliers_x = outliers_identify(x_field, whisker_1, whisker_2)
print("|t 2d. The outliers of x: {}".format(outliers_x))

outliers_x_group0 = outliers_identify(x_list_group0, whisker_1_group0, whisker_2_group0)
print("The outliers of x in group0: {}".format(outliers_x_group0))

outliers_x_group1 = outliers_identify(x_list_group1, whisker_1_group1, whisker_2_group1)
print("The outliers of x in group1: {}".format(outliers_x_group1))

