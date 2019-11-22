# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 13:35:51 2019

@author: sfg17
"""

def Lagrange_poly(x,x_0,x_1,x_2,func):
    c_0 = (x-x_1)*(x-x_2)/ (x_0 - x_1)*(x_0 - x_2)
    c_1 = (x-x_0)*(x-x_2)/ (x_1 - x_0)*(x_1 - x_2)
    c_2 = (x-x_0)*(x-x_1)/ (x_2 - x_0)*(x_2 - x_1)
    
    y_0 = func(x_0)
    y_1 = func(x_1)
    y_2 = func(x_2)
    
    P_2 = c_0 * y_0 + c_1 * y_1 + c_2 * y_2
    
    return P_2

def x_3 ():
    
    num = (x_2**2-x_1**2)*y_0 + (x_0**2 -x_2**2)*y_1 + (x_1**2-x_0**2)*y_2
    denom = (x_2-x_1)*y_0 + (x_0 -x_2)*y_1 + (x_1-x_0)*y_2
    
    x_3 = 1/2 * num/denom
    
    return

def parabolic(x):
    return x*x

if __name__ == "__main__":
    