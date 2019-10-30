# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 13:04:57 2019

@author: sfg17
"""

import numpy as np
import matplotlib.pyplot as plt
from two_Crout import Crout,Forward_Backward


def Linear(x,y,N_density):
    """
    Forms Linear plot by creating a linear regression between each point, 
        and subsequent relevant points. [ y = mx + c ]
    
    ----------
    Parameters:
    ----------
    N_density: Number of extra data points, between each point. 
    x,y: Input arrays of x and y values. 
    """
    x_values = []
    y_values = []
    
    for i in range(len(x)-1):
        m = (y[i+1] - y[i])/ (x[i+1] - x[i])
        c = y[i] - m * x[i]
        
        for j in range(N_density):
            x_tmp = np.linspace(x[i],x[i+1],N_density)
            y_tmp = m * x_tmp + c
            
            x_values.append(list(x_tmp))
            y_values.append(list(y_tmp))
            
    # Flattens List
    x_values = [item for sublist in x_values for item in sublist]
    y_values = [item for sublist in y_values for item in sublist]

    plt.figure()
    plt.plot(x,y,label = "Original Points")
    plt.plot(x_values,y_values,label = "Linear Interpolation")
    plt.legend()
    plt.title("Linear Plot")
    return 

#%%
def Form_Matrix(x,y):
    """
    Forms Matrix of coefficients, corresponding to the 2nd derivatives of the
        function, for the number of points.
        
    If there are N points, we will need N+1 equations to solve, but we will
        only obtain N-1 equations. In this nature, the natural spline boundary
        condition was used. 
        
        --> In applying this, I removed the first and last column of the
            M-2 * M matrix, in order to solve using crouts method. These 
            variables are tehn added back later to solve for the 2nd 
            derivatives
            
    Boundary Conditions: f"o = f"n = 0 
    -----------------------------------

    ----------
    Parameters:
    ----------
    x,y: Input arrays of x and y values. 
    
    """
    N_samples = len(x) 
    matrix = []
    B  = []
    for i in range(1,N_samples-1):
#        print(i)
        row = [0]*N_samples
        row[i-1] = (x[i] - x[i-1])/6
        row[i] = (x[i+1] - x[i-1])/3
        row[i+1] = (x[i+1] - x[i])/6
        b = (y[i+1] - y[i]) / (x[i+1] - x[i]) - (y[i] - y[i-1]) / (x[i] - x[i-1])
        
        matrix.append(row)        
        B.append(b)
        
    matrix[0][0] = 0
    matrix[N_samples-3][N_samples-1] = 0# -1 as len(x) = 4, but 0,1,2,3
    print(np.mat(matrix),B)
    
    # Apply Boundary Condition here, allowing use to delete the 1st and last col
    matrix = np.delete(matrix,0,1)
    matrix = np.delete(matrix,-1,1)

    lower_b,upper_b = Crout(matrix,det = False)
    second_dev = Forward_Backward(lower_b,upper_b,B)
    
    # Including Boundary Conditions Back in
    second_dev.append(0)
    second_dev.insert(0,0)    

    return second_dev



def Cubic_Spline(x,x_i,y_i,title):
    """
    For an input value x, find its closest two x_i points, in order to fit
        it with its suitable cubic spline values. If it is equal to a lower
        bound of x_i, this function selects the next x_i value to perform
        the cubic spline interpolation. 
        
    For example, if x = 3.5, out of x_i = [1,2,3,4,5], we pick x_i = {3,4} 
        in order to perform the interpolations.
        
        orif x = 3, we pick x_i = {3,4} also. 
    
    ----------
    Parameters:
    ----------
    x: array of values to fit using cubic spline interpolation 
    x_i,y_i: array of x,y values used to find the parameters to make cubic 
                spline interpolation
    title: title of graph plot
    
    """
    
        # range to -1 as goes from i to i+1
    def Interpolate(xi_bottom,xi_top,x,f_dash):
        
        A = (xi_top - x)/(xi_top - xi_bottom)
        B = 1- A
        f_i_bot = y_i[x_i.index(xi_bottom)]
        f_i_top = y_i[x_i.index(xi_top)]
        
        C = 1/6 * (A**3 - A)*(xi_top - xi_bottom)**2
        D = 1/6 * (B**3 - B)*(xi_top - xi_bottom)**2
        f_dash_bot =  f_dash[x_i.index(xi_bottom)]
        f_dash_top =  f_dash[x_i.index(xi_top)]
        
        final = A*f_i_bot + B * f_i_top + C * f_dash_bot + D * f_dash_top
        
        return final
        # print(f_i * A)
    
    
    N_x_values = len(x)    
    N_discrete_values = len(x_i) 
    
    f_dash_values = Form_Matrix(x_i,y_i)
    
    spline_x = []
    spline_y = []
        
    # Selects appropriate range that x will fall into
    for i in range(len(x)):
        for j in range(len(x_i)):
            # Skip if larger than final x_i value, so out of range
            if x[i] >= x_i[-1] or x[i] <= x_i[0]:
                pass
            elif x[i] >= x_i[j] and x[i] <= x_i[j+1]:
                
                xi_bottom = x_i[j]
                xi_top = x_i[j+1]
                y_value = Interpolate(xi_bottom,xi_top,x[i],f_dash_values)
                
                spline_x.append(x[i])
                spline_y.append(y_value)

    plt.figure()
    plt.title(title)
    plt.plot(x_i,y_i,'x',label = "Original plots")
    plt.plot(x_i,y_i,label = "Original plots")

    plt.plot(spline_x,spline_y,label = "Spline")
    plt.legend()
 

if __name__ == "__main__":
    
    # Sample data to verify validity of cubic spline plot
    sample_data_x = [1,2,3,5,8,10,18,20,25,45]
    sample_data_y = [4,7,9,12,15,1,3,4,5,3]
    
    # Values given in assignment
    x_i = [-2.1,-1.45,-1.3,-0.2,0.1,0.15,0.9,1.1,1.5,2.8,3.8]
    y_i = [0.012155,0.122151,0.184520,0.960789,0.990050,0.977751,0.422383,0.298197,
         0.105399,3.936690e-4,5.355348e-7]
    
#    print(vals)
    
    x = np.arange(min(sample_data_x),max(sample_data_x),0.001)
    Cubic_Spline(x,sample_data_x,sample_data_y,"Spline with Sample Data")
    
    N_points_between_plots = 5    
    Linear(x_i,y_i,N_points_between_plots)
    
    x = np.arange(min(x_i),max(x_i),0.001)
    Cubic_Spline(x,x_i,y_i,"Spline with Data from Assignment")
    






