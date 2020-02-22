#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 15:34:08 2019

@author: yuqi
"""
import pandas
import matplotlib.pyplot as plt
import numpy as np
import math

GroceriesData = pandas.read_csv('Groceries.csv', delimiter = ',')

nLine = GroceriesData.shape[0]
nCustomer = GroceriesData.iloc[-1:,0:1].Customer.values[0]
print("|t 2a. The number of customers in this market basket data: {}".format(nCustomer))

itemGroup = GroceriesData.groupby(['Item']).count()
print("|t 2b. The number of unique items in the market basket data: {}".format(itemGroup.shape[0]))

CustomerGroup = GroceriesData.groupby(['Customer']).count()
CustomerGroupData = pandas.DataFrame(CustomerGroup)
CustomerGroupData.reset_index(inplace=True)

Item_field = CustomerGroupData.Item.values

min_item = min(Item_field)
max_item = max(Item_field)
item_range_value = max_item - min_item

q1, q2, q3, q4, q5 = np.percentile(Item_field, [0, 25, 50, 75, 100])

iqr = q4 - q2

N_pow = math.pow(9835, -1/3)

h = 2 * iqr * N_pow

print("|t 2c. The dataset which contains the number of distinct items in each customer's market basket:")
print(CustomerGroupData)
print("Histogram of the number of unique items:")
print("The median, the 25th percentile and the 75th percentile in this histogram: {} {} {}".format(q3, q2, q4))
# Visualize the histogram of the field Item
CustomerGroupData.hist(column='Item', bins = int(item_range_value/h) + 1)
plt.title("Histogram of the number of unique items")
plt.xlabel("The number of unique items")
plt.ylabel("The number of customers")
plt.show()

nItemPerCustomer = GroceriesData.groupby(['Customer'])['Item'].count()

freqTable = pandas.value_counts(nItemPerCustomer).reset_index()
freqTable.columns = ['Item', 'Frequency']
freqTable = freqTable.sort_values(by = ['Item'])

nItemPerCustomer.describe()

# Convert the data to the Item List format
ListItem = GroceriesData.groupby(['Customer'])['Item'].apply(list).values.tolist()

# Convert the Item List format to the Item Indicator format
from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit(ListItem).transform(ListItem)
ItemIndicator = pandas.DataFrame(te_ary, columns=te.columns_)

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Find the frequent itemsets
frequent_itemsets = apriori(ItemIndicator, min_support = 75/9835, max_len = 10, use_colnames = True)

print("|t 2d. The k-itemsets which appeared in the market baskets of at least seventy five (75) customers: {}".format(frequent_itemsets))
print("The number of itemsets have found: {}".format(frequent_itemsets.shape[0]))

ItemForzenSet = pandas.DataFrame(frequent_itemsets.itemsets)
HighestK =  ItemForzenSet.iloc[-1:,0:1].itemsets.values[0]
nHighestK = len(HighestK)

print("The highest k value: {}".format(nHighestK))

# Discover the association rules
assoc_rules = association_rules(frequent_itemsets, metric = "confidence", min_threshold = 0.01)
print("|t 2e. The association rules whose Confidence metrics are at least 1%: {} ".format(assoc_rules))
print("The number of association rules have found: {}".format(assoc_rules.shape[0]))

print("|t 2f. The graph:")
import matplotlib.pyplot as plt
plt.figure(figsize=(6,4))
plt.scatter(assoc_rules['confidence'], assoc_rules['support'], s = assoc_rules['lift'])
plt.grid(True)
plt.xlabel("Confidence")
plt.ylabel("Support")
plt.show()

assoc_rules = association_rules(frequent_itemsets, metric = "confidence", min_threshold = 0.6)
print("|t 2g. The rules whoes Confidence metrics are at least 60%: \n")
print(assoc_rules)


