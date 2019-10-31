# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 11:58:47 2019

@author: sfg17
"""

import numpy as np
from functools import reduce

def Crout(matrix,det = True):
    """
    Applies Crout's algorithm to decompoase the matrix into lower and upper
        matrices
    
    ----------
    Parameters:
    ----------
    matrix : represents an input matrix
    det : True/ False determines whether or not to calculate the determinant
    """
    A = matrix

    N_rows = len(A)
    N_columns = len(A[0])

    # Creates a square matrix of zeros
    upper = [[0]*N_columns for i in range(N_rows)]
    lower = [[0]*N_columns for i in range(N_rows)]
    
    # Forms lower and upper matrix 
    for i in range(N_rows):
        for j in range(N_columns):
            
            # Sets diagonal of lower matrix to 1 
            if i ==j:
                lower[i][j] = 1
                
            # Applies iterative equation that updates the upper matrix
            if i <= j:
                rest_sum = np.sum([lower[i][k] * upper[k][j] for k in range(i)])
                upper[i][j] = A[i][j] - rest_sum
                
            # Applies iterative equation that updates the lower matrix
            elif i > j:
                rest_sum = np.sum([lower[i][k] * upper[k][j] for k in range(i-1)])
                lower[i][j] = (A[i][j] - rest_sum)/ (upper[j][j])
                
            else:
                raise ValueError ("Matrix values incorrect, please check")
    
    
    print("a) Original Matrix \n{}\n\n Lower Matrix \n{}\n\n Upper Matrix \n{}\n".format(np.mat(A),np.mat(lower),
                                                                                        np.mat(upper)))
    if det == True:
        
        # Multiplies all the elements in the diagonal of the matrix. 
        det_lower = reduce(lambda x, y: x*y, [lower[i][i]  for i in range(len(lower))])
        det_upper = reduce(lambda x, y: x*y, [upper[i][i]  for i in range(len(upper))])
        det_A = det_lower*det_upper
        
        print("b) Det(Lower) = {}, Det(Upper) = {}, Det(A) = {}, Floating point error = {}\n".format(det_lower,det_upper,det_A,
                                                                                         det_A - det_lower*det_upper))
    else:
        pass
    
    return lower,upper

def Forward_Backward(lower,upper,B = None):
    """
    Performs a forward transformation on the lower matrix, and a backward 
    transformation on the upper matrix
    
    Ax = LUx = b
    
    L(y) = b where y = (Ux)
    
        solve for y first, then solve for x.
    """
    L = lower
    U = upper
    
    N_rows = len(L)

    # Forwards
    y_values = []
    for i in range(N_rows):
        sum_rest = sum([L[i][k] * y_values[k] for k in range(len(y_values))])
        y_m = (B[i] - sum_rest)/ L[i][i]
        y_values.append(y_m)        
    # print(y_values)

    # Backwards
    x_values = [0]*N_rows
    for i in range(N_rows-1,-1,-1):
        sum_rest = sum([U[i][j] * x_values[j] for j in range(i+1,N_rows)])
        x_values[i] = (y_values[i] - sum_rest) / U[i][i]

    print("c)d) Value of x = {}\n".format(x_values))
    
    return x_values

def Inverse(matrix,lower,upper):
    """
    Finds the inverse of the matrix, by finding inverse of the upper and lower,
        and multiplying the result of the two. 
    
    

    A^-1 = B^-1 * C^-1
    """
    U_inverse = np.linalg.inv(upper)
    L_inverse = np.linalg.inv(lower)
    inverse_mat = np.matmul(U_inverse,L_inverse)

    print("e) Inversed Matrix",inverse_mat)
    # print(np.linalg.inv(matrix))

if __name__ == "__main__":
    # mat_b = [[5,4,1],[10,9,4],[10,13,15]]
    mat_b = [[3,1,0,0,0],
             [3,9,4,0,0],
             [0,9,20,10,0],
             [0,0,-22,31,-25],
             [0,0,0,-55,61]]
    
#    mat_b = [[1,2],[3,4]]

    B = [2,5,-4,8,9]
    
    lower_b, upper_b = Crout(mat_b,det = True)
    Forward_Backward(lower_b,upper_b,B)
    Inverse(mat_b,lower_b,upper_b)
