# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 13:04:57 2019

@author: sfg17
"""

import numpy as np
import matplotlib.pyplot as plt
from two_Crout import Crout,Forward_Backward

x = [-2.1,-1.45,-1.3,-0.2,0.1,0.15,0.9,1.1,1.5,2.8,3.8]
y = [0.012155,0.122151,0.184520,0.960789,0.990050,0.977751,0.422383,0.298197,
     0.105399,3.936690e-4,5.355348e-7]

#%% Linear Plot

def Linear(x,y):
    plt.plot(x,y)
    plt.plot(x,y,'x')
    
    plt.title("Linear Plot")
    return

Linear(x,y)

#%%



def Form_Matrix(x,y):
    """
    Boundary Conditions: f"o = f"n = 0
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

sample_data_x = [1,2,3,5,8]
sample_data_y = [4,7,9,12,15]
vals = Form_Matrix(sample_data_x,sample_data_y)

print(vals)

        
#%%\

def Cubic_Spline(x,x_i,y_i):
    """
    For all the x values input, calculate the f(x) for all the x_i and f_is,
        then iterate through to repeat through all of them
    
    """
    N_x_values = len(x)    
    N_discrete_values = len(x_i) 
    
    f_dash = Form_Matrix(x_i,y_i)
    f_dash = 
    
    # range to -1 as goes from i to i+1
    def Interpolate(xi_bottom,xi_top,x):
        
        A = (xi_top - x)/(xi_top - xi_bottom)
        B = 1- A
        f_i_bot = y_i[x_i.index(x_bottom)]
        f_i_top = y_i[x_i.index(x_top) + 1]
        
        C = 1/6 * (A**3 - A)*(xi_top - x_bottom)**2
        D = 1/6 * (B**3 - B)*(xi_top - x_bottom)**2
        f_dash_bot =  y_i[x_i.index(x_bottom)]
        print(f_i * A)
#        f = A*y_i[]

x = np.arange(0,10,1)
Cubic_Spline(x,sample_data_x,sample_data_y)
            







