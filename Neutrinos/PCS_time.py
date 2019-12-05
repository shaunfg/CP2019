#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 22:35:29 2019

@author: ShaunGan
"""

import numpy as np

def PCS(T_o,T_f,T_step,i):
    """
    probabilistic cooling scheme
    """
    N = abs(T_f - T_o) / T_step
    PE = 0.3
    PL = 0.29
    a = 0.01
    
    A = (T_o - T_f)*(N+1)/N
    B = T_o - A
    T_i = (A/(i+1)+B)*PE + (a*T_o/np.log(1+i))*PL
    
#    print(i,N)
    
    return T_i

t = 1
count = 0
while t >0:

    count +=1
    i = count
    t = PCS(100,0,1,i)
    print(t)

#    if i ==1:
#        print(t)