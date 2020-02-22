#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 15:05:53 2019

@author: yuqi
"""
import numpy as np
from itertools import combinations

nItemSets = 2**7 - 1

alphabet_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

print("|t 1a. The number of possible itemsets: {}\n".format(nItemSets))
ItemSets1 = list(combinations(alphabet_list, 1))
print("|t 1b. All the possible 1-itemsets: {}\n".format(ItemSets1))

ItemSets2 = list(combinations(alphabet_list, 2))
print("|t 1c. All the possible 2-itemsets: {}\n".format(ItemSets2))

ItemSets3 = list(combinations(alphabet_list, 3))
print("|t 1d. All the possible 3-itemsets: {}\n".format(ItemSets3))

ItemSets4 = list(combinations(alphabet_list, 4))
print("|t 1e. All the possible 4-itemsets: {}\n".format(ItemSets4))

ItemSets5 = list(combinations(alphabet_list, 5))
print("|t 1f. All the possible 5-itemsets: {}\n".format(ItemSets5))

ItemSets6 = list(combinations(alphabet_list, 6))
print("|t 1g. All the possible 6-itemsets: {}\n".format(ItemSets6))

ItemSets7 = list(combinations(alphabet_list, 7))
print("|t 1h. All the possible 7-itemsets: {}\n".format(ItemSets7))