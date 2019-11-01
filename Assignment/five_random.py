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
#%% Uniform Random Variables

def Random_Uniform():
    """
    Generates Random variables in a uniform PDF.
    """
    # Generate random variables using in-built random function
    random_variables = [random.uniform(0,1) for i in range(10**5)]
    
    # Generate random variables using  random function
    random_variables_np = np.random.random(10**5)
    
    # Plots values
    fig, (ax1,ax2) = plt.subplots(2,1, figsize = (5,8))
    ax1.set_title("In-built random function - Uniform")
    ax1.hist(random_variables,bins = 50,color = 'blue')
    ax2.set_title("Numpy random function - Uniform")
    ax2.hist(random_variables_np,bins = 50,color = 'orange')
    
    return

#%% Transformation Method


def Transformation():
    """
    Applying the transformation method to a invertible function, in order to 
        generate random distribution, following a predetermines PDF function.
    
    (Function is already hard-coded)
    -----------
    Working out 
    -----------
    In this case:
        PDF(x) = 1/2 * cos(x/2)
    and thus: 
        F(x) = Integrate: PDF(x) = sin(x/2)
    Since F(x) = x'
        x = 2 * arcsin(x').
    Thereore, sub values into x' to get values for the new PDF.
    """
    
    start_time = timeit.default_timer()

    # Apply transformation to random variables.
    random_variables = [random.uniform(0,1) for i in range(10**5)]
    transformation = np.arcsin(random_variables) * 2
    
    elapsed_time = timeit.default_timer() - start_time_transformation
 
    verify_x = np.linspace(0,np.pi,100)
    
    # Plots the histogram of distribution, and PDF to verify.
    plt.figure()    
    heights, bins, patches = plt.hist(transformation,bins = 50,
                                      label = "Generated points")
    plt.plot(verify_x, 1/2 * np.cos(verify_x/2) * 2 * heights[0],
             label = "PDF")
    plt.title("Histogram of random points using transformation method")
    plt.legend()
    return start_time, elapsed_time
    
    
#%% Rejection Method

def Rejection_Method(PDF_function,comparison_function, x_lim,title,scatter):
    """
    For a function that is not invertible, apply the rejection method to 
        estimate a distribution that follows a similar shape. Works by
        randomly accumulating data points fit a criteria between the PDF
        and the comparison function
    
    ---------    
    Criteria:
    ---------
    
        Accept if: PDF(x_i) > p_i
        
    where PDF is the pdf function, x_i is a randomly generated x value
    from 0 to the limit, and p_i is a randomly generated y value, between
    0 and the comparison function 
        
    -----------
    Parameters:
    -----------
    
    PDF_function: represents the PDF function to fit
    comparison_function: represents the comparison function
    x_lim: sets the range from 0 on the x axis
    title: sets the type of comparison function being used
    scatter: True/False, plots the scatter function (only implemented for 
                    uniform currently.)
    
    """
    start_time= timeit.default_timer()

    # Prepares arrays
    x_values = []
    x_values_rejected = []
    
    y_values = []
    y_values_rejected = []
    
    while len(x_values) < 10**5:
        
        # Generates PDF(y) for a random x value
        random_PDF_x = random.uniform(0,x_lim)
        random_PDF_y = PDF_function(random_PDF_x)
        
        # Generated a random y value, between 0 and the comparison function: p_i
        random_comp_x = random.uniform(0,x_lim)
        random_comp_y = random.uniform(0,comparison_function(random_comp_x))

        # Accept if PDF(x_i) is larger than p_i
        if random_PDF_y > random_comp_y:
            x_values.append(random_PDF_x)
            y_values.append(random_comp_y)
        
        # Reject if else. 
        else:
            x_values_rejected.append(random_PDF_x)
            y_values_rejected.append(random_comp_y)
            pass
        
    elapsed_time = timeit.default_timer() - start_time

    # Plot graphs to display results
    
    # Reduce on computation time
    N_plots = 5000

    fig,(ax1,ax2) = plt.subplots(2,1,figsize=(7,8))
    
    sample_x = np.linspace(0,np.pi,1000)
    heights,bins,patched = ax1.hist(x_values,bins = 100,
                                    label = "Rejection Generated Points") 
    ax1.plot(sample_x,PDF_function(sample_x)*heights[0] *np.pi/2,
             label = "PDF")
#    ax1.plot(sample_x,comparison_function(sample_x)*heights[0] *np.pi/2,
#             label = "PDF")    
    verify_x = np.linspace(0,x_lim,100)
    ax2.plot(verify_x,comparison_function(verify_x),label = 'Comparison')
    ax2.plot(verify_x,PDF_function(verify_x),label = 'Original')

    ax1.set_title("Histogram of random points using {}".format(title))
    ax2.set_title("PDF vs Comparison Function")
    ax1.legend()
    ax2.legend()
    
    if scatter == True:
        plt.figure()
        plt.plot(x_values[:N_plots],y_values[:N_plots],"x",label = "Accepted")    
        plt.plot(x_values_rejected[:N_plots],y_values_rejected[:N_plots],"x",
                 label = "Rejected")
        plt.title("Scatter plot for {} function".format(title))
        plt.legend()

    else:
        pass
    
    Efficiency = len(x_values)/ (len(x_values_rejected)+len(x_values)) * 100
    
    print("Efficiency of operation for {} = {:.2f}%".format(title,Efficiency))

    return start_time, elapsed_time

# Uniform function, returns constant for all x
def func_uniform(x):
    return x*0 +0.65

# Comparison function 
def func_comparison_example(x):
    return 3/10 * np.cos(x) + 0.4

# Probability density function 
def PDF(x):
    return 2/np.pi * np.cos(x/2)**2



if __name__ == "__main__":
    """
    Start times were returned from transformation and rejection method 
        function, so that only the variable computation time would be
        considered. 
        
    """
    
    # Set limit for x range    
    x_lim = np.pi
    
    # Runs Random Uniform generation
    Random_Uniform()   
    
    # Runs Transformation Method
    start_time_transformation, elapsed_transformation = Transformation()

    # Runs Rejection method for cos comparison 
    comparison_type = "cos comparison"
    start_cos, elapsed_cos = Rejection_Method(PDF,func_comparison_example,
                                              x_lim,comparison_type,False)

    # Runs Rejection method for uniform comparison     
    comparison_type = "uniform comparison"
    start_uni, elapsed_uni = Rejection_Method(PDF,func_uniform,x_lim,
                                              comparison_type,True)
    
    # Calcullates ratio of rejection methods to transformation methods
    ratio_uni = elapsed_transformation/elapsed_uni
    ratio_cos = elapsed_transformation/elapsed_cos

    # Display results
    print("Transformation Method is {:.2f} times slower than rejection for {}\n".format(
            ratio_cos,comparison_type))
    
    print("Transformation Method is {:.2f} times slower than rejection for {}\n".format(
            ratio_uni,comparison_type))
    
    print("cos comparison is x{:.2f} times faster than uniform comparison".format(
            elapsed_cos/elapsed_uni))
    
    
    plt.show()
    
    
