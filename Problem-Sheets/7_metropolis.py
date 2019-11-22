#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 23:22:48 2019

@author: ShaunGan
"""

import numpy as np
import matplotlib.pyplot as plt
import random

def Gaussian_gen(mean):
    # Returns list of gaussian values, centered on mean 
    x = np.arange(-5,5,0.05)
    sigma = 1
    PDF = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-1/2*((x-mean)/sigma)**2)
    
    return PDF

def Gaussian(x,mean):
    # Returns list of gaussian values, centered on mean 
    sigma = 1
    PDF = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-1/2*((x-mean)/sigma)**2)
    return PDF

def Metropolis(PDF):
    
    p_values = []

    x_values = []
    x = random.uniform(0,1)

    for y in range(1000):
        # Sample random x
        gauss_samples = Gaussian_gen(x)
        
        # Picks random value near input x
        x_dash = random.choice(gauss_samples)

        if PDF(x_dash,x) >= PDF(x,x):
            p = 1
            x = x_dash
        else:
            p = PDF(x_dash,x)/PDF(x,x)
            
            
        p_values.append(p)
        x_values.append(x_dash)
    
     
    # Shows that the points sampled, are very close to the that mean value
    # and hence increases  efficiency! 
    plt.hist(p_values,bins = 30)
    plt.figure()
    plt.hist(x_values,bins = 30)
    
    print(x_values)
        
        
    
Metropolis(Gaussian)