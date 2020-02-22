import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

def density_estimator(x_list, x_min, x_max, h):
    value = x_min
    m = []
    mm = []
    result_list = []
    value = value + h/2.0
    m.append(value)
    while value <=x_max:
        value = value + h
        m.append(value)
    for m1 in m:
        x_in_or_not = 0
        mm.append(m1)
        for x in x_list:
            if x - m1 > -h/2.0 and x - m1 <= h/2.0:
                x_in_or_not = x_in_or_not + 1
        density = x_in_or_not/(len(x_list)*h)
        result_list.append(density)
        mm.append(m1 + h/2.0)
        result_list.append(density)
        #print("(", round(m1, 4), ",", density, ")")
    d = {'x': mm, 'y': result_list}
    df = pd.DataFrame(data=d)
    return df


# Read in data
trainData = pd.read_csv('NormalSample.csv')

# Remove all missing observations
trainData = trainData.dropna()
# print(trainData)

x_field = trainData.x.values

print("|t 1d. The coordinates of the density estimator(h=0.1): ")
df1 = density_estimator(x_field, 26, 36, 0.1)
#The graph of Question 1d
#Visualize the histogram of a b and h=0.1
plt.plot(df1.x, df1.y, linestyle="steps-pre", label='h = 0.1')
plt.xticks(np.arange(26, 36, 1))
plt.legend()
plt.grid()
plt.show()

print("|t 1e. The coordinates of the density estimator(h=0.5): ")
df2 = density_estimator(x_field, 26, 36, 0.5)
#The graph of Question 1e
#Visualize the histogram of a b and h=0.5
plt.plot(df2.x, df2.y, linestyle="steps-pre", label='h = 0.5')
plt.xticks(np.arange(26, 36, 1))
plt.legend()
plt.grid()
plt.show()


print("|t 1f. The coordinates of the density estimator(h=1): ")
df3 = density_estimator(x_field, 26, 36, 1)
#The graph of Question 1f
#Visualize the histogram of a b and h=1
plt.plot(df3.x, df3.y, linestyle="steps-pre", label='h = 1')
plt.xticks(np.arange(26, 36, 1))
plt.legend()
plt.grid()
plt.show()

print("|t 1g. The coordinates of the density estimator(h=2): ")
df4 = density_estimator(x_field, 26, 36, 2)
#The graph of Question 1g
#Visualize the histogram of a b and h=2
plt.plot(df4.x, df4.y, linestyle="steps-pre", label='h = 2')
plt.xticks(np.arange(26, 36, 1))
plt.legend()
plt.grid()
plt.show()

