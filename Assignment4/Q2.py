#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 20:32:52 2019

@author: yuqi
"""

def perceptronOr(x1,x2):
    weight1 = 1
    weight2 = 1
    c = 1
    if (weight1*x1 + weight2*x2 >= c):
        return 1
    else:
        return 0

    
def perceptronAnd(x1,x2):
    weight1 = 1
    weight2 = 1
    c = 2
    if (weight1*x1 + weight2*x2 >= c):
        return 1
    else:
        return 0

def perceptronXAnd(x1, x2):
    weight11 = -1
    weight12 = 1
    c1 = 0
    
    weight21 = 1
    weight22 = -1
    c2 = 0
    
    weight31 = 1
    weight32 = 1
    c3 = 2
    
    if (weight11*x1 + weight21*x2 >= c1):
        h1 = 1
    else:
        h1 = 0
        
    if (weight12*x1 + weight22*x2 >= c2):
        h2 = 1
    else:
        h2 = 0
    
    if (weight31*h1 + weight32*h2 >= c3):
        return 1
    else:
        return 0

print('|t 2a. To implement the logical OR function: w1 = 1 w2 = 1 c = 1')

print("Call perceptronOr(x1 = 0, x2 = 0), return: {}".format(perceptronOr(0, 0)))
print("Call perceptronOr(x1 = 1, x2 = 0), return: {}".format(perceptronOr(1, 0)))
print("Call perceptronOr(x1 = 0, x2 = 1), return: {}".format(perceptronOr(0, 1)))
print("Call perceptronOr(x1 = 1, x2 = 1), return: {}".format(perceptronOr(1, 1)))

print('\n|t 2b. To implement the logical AND function: w1 = 1 w2 = 1 c = 2')
print("Call perceptronAnd(x1 = 0, x2 = 0), return: {}".format(perceptronAnd(0, 0)))
print("Call perceptronAnd(x1 = 1, x2 = 0), return: {}".format(perceptronAnd(1, 0)))
print("Call perceptronAnd(x1 = 0, x2 = 1), return: {}".format(perceptronAnd(0, 1)))
print("Call perceptronAnd(x1 = 1, x2 = 1), return: {}".format(perceptronAnd(1, 1)))

print('\n|t 2b. To implement the logical XAND function: w11= -1 w21 = 1 c1 = 0 ')
print(' w12 = 1  w22 = -1  c2 = 0')
print(' w31 = 1   w32 = 1    c3 = 2')
print("Call perceptronXAnd(x1 = 0, x2 = 0), return: {}".format(perceptronXAnd(0, 0)))
print("Call perceptronXAnd(x1 = 1, x2 = 0), return: {}".format(perceptronXAnd(1, 0)))
print("Call perceptronXAnd(x1 = 0, x2 = 1), return: {}".format(perceptronXAnd(0, 1)))
print("Call perceptronXAnd(x1 = 1, x2 = 1), return: {}".format(perceptronXAnd(1, 1)))
