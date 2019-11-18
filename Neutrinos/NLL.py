#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 20:25:46 2019

@author: ShaunGan
"""

import numpy as np

def NLL(rate,N_observed_events):
    l = rate
    k = N_observed_events
    
    tmp = []
    for i in range(len(rate)):
        
        if k[i] == 0:
            pass
        else:       
            value = l[i] - k[i] + k[i] * np.log(k[i]/l[i])
            tmp.append(value)
        
    NLL_value = sum(tmp)
    
    return NLL_value
        
