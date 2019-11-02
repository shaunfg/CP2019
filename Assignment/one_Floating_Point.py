# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 11:40:47 2019

@author: sfg17
"""

import numpy as np

def machineEps(float_type):
    """
    Finds the machine epsilon, based on theoretical equation.
    
        E = base ^ -(precision - 1)
    
    where E is the machine epsilon, base is 2 for binary and precision is based
    on the length of the mantissa. 
    ----------
    Parameters:
    ----------
    float_type : inputs the type of float function (e.g. float32, float64)
    """
    base = 2
    
    if float_type == "32bit":
        # One implicit bit
        precision = 24
    elif float_type == "64bit":
        # One implicit bit
        precision = 53
    elif float_type == "extended":
        precision = 64
    
    # Finds machine epsilon for each float type
    eps = base**(-(precision-1))
    eps_abs_rounding = eps/2
    
    print("CHECK1: {}, Machine Epsilon = {}, \n(Machine Epsilon Rounding = {})\n".format(
            float_type,eps,eps_abs_rounding))
    return

def machineEps_check(float_type):
    """
    Verifies the machine epsilon of local hardware, by finding the value in
        which the the float of a value, is no longer equal to itself.
        This occurs when the value, or machine epsilon, is smaller than the 
        precision that can be obtained by the float. Falsely displaying the
        floats as equal.
        
    This was obtained through the while loop found below

    ----------
    Parameters:
    ---------2
    float_type : inputs the type of float function (e.g. float32, float64)    
    
    """
    
    machine_eps = float_type(1)
    while float_type(1) + float_type(machine_eps) != float_type(1):
        machine_eps_last = machine_eps
        machine_eps = float_type(machine_eps) / float_type(2)
                
        
    print("{}\nMachine Epsilon = {},\n(Machine Epsilon Rounding = {})\n".format(
            float_type,machine_eps_last,machine_eps))
    return 





if __name__ == "__main__":
    print("----------- Hardware Machine Epsilon --------------\n")
    
    for float_type in [np.float32,np.float64]:
        machineEps_check(float_type)
        
    print("----------- Theoretical Machine Epsilon ------------\n")

    for float_type in ["32bit","64bit","extended"]:
        machineEps(float_type)
    

    
    print("------------- Numpy Machine Epsilon ---------------\n")
    
    print("CHECK2: Machine Epsilon = ",np.finfo(float).eps,"\n")

    print("---------------------------------------------------\n")
