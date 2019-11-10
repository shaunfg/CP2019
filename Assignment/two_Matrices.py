# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 11:58:47 2019

@author: sfg17
"""

import numpy as np
from functools import reduce

def Crout(matrix,det = True):
    """
    Applies Crout's algorithm to decompose the matrix into lower and upper
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
    output = [[0]*N_columns for i in range(N_rows)]
    
    # Forms lower and upper matrix 
    for i in range(N_rows):
        for j in range(N_columns):
            
            # Sets diagonal of lower matrix to 1 
            if i ==j:
                lower[i][j] = 1
                
            # Applies iterative equation that updates the upper matrix
            if i <= j:
                # rest_sum first calculates the sum component of the equation
                rest_sum = np.sum([lower[i][k] * upper[k][j] for k in range(i)])
                upper[i][j] = A[i][j] - rest_sum
                
                output[i][j] = upper[i][j]
                
            # Applies iterative equation that updates the lower matrix
            elif i > j:
                rest_sum = np.sum([lower[i][k] * upper[k][j] for k in range(i)])
                lower[i][j] = (A[i][j] - rest_sum)/ (upper[j][j])

                output[i][j] = lower[i][j]

            else:
                raise ValueError ("Matrix values incorrect, please check")
    
    
    print("a) Original Matrix \n{}\n\nLower Matrix \n{}\n\nUpper Matrix \n{}\n".format(np.mat(A),np.mat(lower),
                                                                                        np.mat(upper)))
    print("Combined Matrix\n{}\n".format(np.mat(output)))
    
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
        transformation on the upper matrix. This is to solve and equation
        of the form: 
            
        Ax = LUx = b
            
        where A is the original matrix, L is lower, U is upper, x is the
        variables we solve for and b is a vector of constants.
    
    First, let y = (Ux), so:
        L(Ux) = L(y) = b
    and solve for y, then :
        y = Ux
    and solve for x.
    -----------
    Parameters:
    -----------
    lower: lower matrix
    upper: upper matrix
    B: Solutions to the matrix equation
    """
    L = lower
    U = upper
    
    N_rows = len(L)

    # Forwards
    y_values = [] # location to store solutions
    for i in range(N_rows):
        # Applies equation to calculate solutions to forward
        rest_sum = sum([L[i][k] * y_values[k] for k in range(i)])
        y_m = (B[i] - rest_sum)/ L[i][i]
        y_values.append(y_m)        

    # Backwards
    x_values = [0]*N_rows # location to store solutions
    for i in range(N_rows-1,-1,-1): # goes from bottom to top
        # Applies equation to calculate solutions to forward
        rest_sum = sum([U[i][j] * x_values[j] for j in range(i+1,N_rows)])
        x_values[i] = (y_values[i] - rest_sum) / U[i][i]

    print("c)d) Value of x = {}\n".format(x_values))
    return x_values

def Inverse(lower,upper):
    """
    Finds the inverse of the matrix, by finding inverse of the upper and lower,
        and multiplying the result of the two, as shown: 

        A^-1 = B^-1 * C^-1
        
        where A,B,C are matrices
    -----------
    Parameters:
    -----------
    matrix: original matrix
    lower: lower matrix
    upper: upper matrix    
    """
    # Finds inverse of lower and upper matrices.
    U_inverse = np.linalg.inv(upper)
    L_inverse = np.linalg.inv(lower)
    
    # Inverse found by multiplying upper and lower
    inverse_mat = np.matmul(U_inverse,L_inverse)

    print("e) Inversed Matrix\n",inverse_mat)

    # Rounded values for appropriate printing
#    rounded_values = [list(np.round(inverse_mat[i],3)) for i in range(len(inverse_mat))]
#    print(np.mat(rounded_values))
    
if __name__ == "__main__":

    mat_b = [[3,1,0,0,0],
             [3,9,4,0,0],
             [0,9,20,10,0],
             [0,0,-22,31,-25],
             [0,0,0,-55,61]]
    
    B = [2,5,-4,8,9]

    # Runs all the functions
    lower_b, upper_b = Crout(mat_b,det = True)
    Forward_Backward(lower_b,upper_b,B)
    Inverse(lower_b,upper_b)
