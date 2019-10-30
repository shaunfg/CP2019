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
verify_x = np.linspace(0,np.pi,100)
plt.plot(verify_x,1/2 * np.cos(verify_x/2))

elapsed_ii = timeit.default_timer() - start_time_ii

#%% Rejection Method

"""
If not invertible function, the use rejection method

"""

def Rejection_Method(PDF_function, comparison_function, x_lim,title):
    
    x_values = []
    x_values_rejected = []
    
    x_values_plot = []
    x_values_plot_rejected = []
    y_values = []
    y_values_rejected = []
    
    all_x = []
    all_y = []

    count = 0
    
    while len(x_values)< 10**5:
        random_PDF_x = random.uniform(0,x_lim)
        random_PDF_y = PDF_function(random_PDF_x)
        
        random_comp_x = random.uniform(0,x_lim)
        random_comp_y = random.uniform(0,comparison_function(random_comp_x))
        
        all_x.append(random_comp_x)
        all_y.append(random_comp_y)

    #    random_comp_y = random.uniform(0,0.65)
        
        if random_PDF_y > random_comp_y:
            x_values.append(random_PDF_x)
            x_values_plot.append(random_comp_x)
            y_values.append(random_comp_y)

        else:
            count+=1
            x_values_rejected.append(random_PDF_x)
            x_values_plot_rejected.append(random_comp_x)
            y_values_rejected.append(random_comp_y)
#            plt.plot(x_values_plot_rejected,y_values_rejected)
            # plt.plot(x_values_rejected,y_values_rejected)
            # plt.show()
            pass
    N_plots = 5000
    verify_x = np.linspace(0,x_lim,100)

    plt.figure()    
    plt.plot(all_x[:5000],all_y[:5000])
    plt.title("thisone")

    plt.figure()
    plt.plot(x_values_plot_rejected[:N_plots],y_values_rejected[:N_plots],'x',alpha = 0.5,)
    plt.plot(x_values_plot[:N_plots],y_values[:N_plots],'x')
    
    fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=(8,13))
    heights,bins,patched = ax1.hist(x_values,bins = 100) 
    sample_x = np.linspace(0,np.pi,1000)
    ax1.plot(sample_x,PDF_function(sample_x)*heights[0] *np.pi/2)
    ax2.plot(verify_x,comparison_function(verify_x),label = 'Comparison')
    ax2.plot(verify_x,PDF_function(verify_x),label = 'Original')
    
    ax3.plot(x_values[:N_plots],y_values[:N_plots],"x",label = "Accepted")    
    ax3.plot(x_values_rejected[:N_plots],y_values_rejected[:N_plots],"x",
             label = "Rejected")
    
    ax1.set_title("Histogram of random points using {}".format(title))
    ax2.set_title("PDF vs Comparison Function")
    ax3.set_title("Scatter plot of accepted and rejected points")
    ax2.legend()
    ax3.legend()
    Efficiency = len(x_values)/ (len(x_values_rejected)+len(x_values))
#    
    print("Efficiency of operation = {}".format(Efficiency))


    return x_values_rejected,y_values_rejected


def func_uniform(x):
    return x*0 +0.65

def func_comparison_example(x):
    return 3/10 * np.cos(x) + 0.5#0.337

def PDF(x):
    return 2/np.pi * np.cos(x/2)**2


x_lim = np.pi

start_time_iii = timeit.default_timer()
Rejection_Method(PDF,func_comparison_example,x_lim,"cos comparison")
elapsed_iii = timeit.default_timer() - start_time_iii
ratio = elapsed_iii/elapsed_ii
print("Rejection Method is {:.2f} times slower".format(ratio))

start_time_iii = timeit.default_timer()
Rejection_Method(PDF,func_uniform,x_lim,"uniform comparison")
elapsed_iii = timeit.default_timer() - start_time_iii
ratio = elapsed_iii/elapsed_ii
print("Rejection Method is {:.2f} times slower".format(ratio))

plt.show()
#TODO Theoretical Ratio?


