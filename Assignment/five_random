#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 16:26:46 2019

@author: ShaunGan
"""

import random
import matplotlib.pyplot as plt
import numpy as np
import timeit


random_variables = [random.uniform(0,1) for i in range(10**5)]
plt.figure()
plt.title("In-built random function")
plt.hist(random_variables,bins = 50,color = 'blue')

random_variables_np = np.random.random(10**5)
plt.figure()
plt.title("Numpy random function")
plt.hist(random_variables_np,bins = 50,color = 'orange')


#%% Transformation Method

"""
PDF(x) = 1/2 * cos(x/2)
F(x) = Integrate: PDF(x) = sin(x/2)

Since F(x) = x'

--> x = 2 * arcsin(x')

(Sub values into x' to get values for the new PDF)
"""
start_time_ii = timeit.default_timer()

transformation = np.arcsin(random_variables) * 2
plt.hist(transformation,bins = 50)

plt.figure()
verify_x = np.linspace(0,1,100)
plt.plot(verify_x,1/2 * np.cos(verify_x/2))

elapsed_ii = timeit.default_timer() - start_time_ii

#%% Rejection Method

"""
If not invertible function, the use rejection method

"""
start_time_iii = timeit.default_timer()


def comparison_function(x):
    return 1/2 * np.cos(x/2)* 2.4 - 0.56

def actual_function(x):
    return 2/np.pi * np.cos(x/2)**2

y_min = 0
y_max = 1

x_values = []
x_values_rejected = []
count = 0

while len(x_values)< 10**5:
    random_actual_x = random.uniform(0,1)
    random_actual_y = actual_function(random_actual_x)
    
    random_comp_x = random.uniform(0,1)
    random_comp_y = comparison_function(random_comp_x)
#    random_comp_y = random.uniform(0,0.65)
    
    if random_actual_y < random_comp_y:
        count+=1
        x_values_rejected.append(random_actual_x)
        pass
    else:
        x_values.append(random_actual_x)
        
plt.figure()
plt.hist(x_values,bins = 100) 

elapsed_iii = timeit.default_timer() - start_time_iii


plt.figure()
plt.plot(verify_x,comparison_function(verify_x),label = 'comparison')
verify_x = np.linspace(0,1,100)
plt.plot(verify_x,actual_function(verify_x))
plt.legend()

ratio = elapsed_iii/elapsed_ii

print("Rejection Method is {:.2f} times slower".format(ratio))

#TODO Theoretical Ratio?

