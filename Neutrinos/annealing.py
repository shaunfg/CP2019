#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 23:22:48 2019

@author: ShaunGan
"""

import numpy as np
import matplotlib.pyplot as plt
import random


#def Gaussian(x,mean):
#    # Returns list of gaussian values, centered on mean 
#    sigma = 1
#    PDF = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-1/2*((x-mean)/sigma)**2)
#    return PDF
#


def Ackley(x,y):
    A = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))
    B = - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
    f = A + B + np.exp(1) + 20
    
    return f

#
#def func(x,m):
#    return (m-5)**2 +(x-3)**2

    
def Simulated_Annealing(func,T_start,T_step,guess_x= [-5,5]):
    
    def Thermal(E,T):
        k_b = 1e-23
        return np.exp(- E / (k_b * T))
    
    x_values = []
    
    x_bot = guess_x[0]
    x_top = guess_x[1]
    

    h,h_2 = (0.05,0.05) # step sizes (initial conditions)
    # Set first guess 
    x = random.uniform(x_bot,x_top)
    y = random.uniform(x_bot,x_top)
    
    T = T_start

    while T > 0:
        x_dash = random.uniform(x-h,x+h)
        y_dash = random.uniform(y-h_2,y+h_2)

        del_f = func(x_dash,y_dash)- func(x,y) # Negative value if approve means that your function has gotten smaller 
        p_acc = Thermal(del_f,T)  # So this must be >= 1 in order for it to make any sense 
        if p_acc > 1  : # if next energy value is smaller, then update            
            h = 1#(x_dash - int(x_dash)) / 2
            h_2 = 1#(y_dash - int(y_dash)) / 2

            x = x_dash
            y = y_dash
        else:
#            print('----',x_dash,x)
            pass 

        x_values.append(x_dash)
        
        T-= T_step
    print(y,x)
    
    plt.plot(x_values,func(np.array(x_values),1),'x')
#    print(x_values)
    plt.figure()
    
    print()
#    plt.hist(x_values,bins = 80)
    
    
def step_size(x,T_s_o,T_o):
    T_s = np.exp(np.log(T_s_o)/T_o * x) - 1
    return T_s
        
sample_x = np.linspace(-5,5,1000)
plt.plot(sample_x, Ackley(sample_x,1))
Simulated_Annealing(Ackley,T_start = 100,T_step = 0.001)

# Notes: Increasing the step size would increase the chance of finding the smallest minima