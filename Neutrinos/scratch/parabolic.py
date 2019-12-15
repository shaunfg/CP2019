# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 13:35:51 2019

@author: sfg17
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import warnings




def Parabolic_minimise(func,guess_x = [-2.2,0.1,2]):
    """
    generate f(x) from a set of x values
    append the new x_3 value
    
    find f(x) with all four values
    save the smallest three values
    """

    def _find_next_point(x_list, y_list):
        """

        accepts list of x and y, each of 3 elements
        """
        x_0, x_1, x_2 = tuple(x_list)
        y_0, y_1, y_2 = tuple(y_list)
        num = (x_2 ** 2 - x_1 ** 2) * y_0 + (x_0 ** 2 - x_2 ** 2) * y_1 + (x_1 ** 2 - x_0 ** 2) * y_2
        denom = (x_2 - x_1) * y_0 + (x_0 - x_2) * y_1 + (x_1 - x_0) * y_2
        if denom != 0:
            x_3 = 1 / 2 * num / denom
        else:
            # if denomiator is zero, pick mid point of x
            x_list = [x for x in x_list if x <= x_list[np.argmax(x_list)]]
            x_list = [x for x in x_list if x >= x_list[np.argmin(x_list)]]
            x_3 = x_list[0]
        return x_3

    random_x = guess_x#[random.uniform(x_bottom,x_top) for x in range(3)]
    random_y = func(np.array(random_x))

#    print(random_x)
    x_3 = _find_next_point(random_x,random_y)
    random_x.append(x_3)
    random_y = func(np.array(random_x))

    limit = 100
    for i in range(limit):
        
        # Find maximum f(x) values
        max_idx = np.argmax(random_y)
        
        # delete the smallest x value
        del random_x[max_idx]
#        print("-",random_y)
                
        # Finds the new f(x) values
        random_y = func(np.array(random_x))

        # Finds the next minimum value
        x_3 = _find_next_point(random_x, random_y)
        print(x_3)

        # Check for negative curvature
        if func(x_3) > all(random_y):
            warnings.warn("Interval has positive & negative curvature", Warning) 
            
            random_x.append(x_3)

            # finds 2 additional values from max and min of interval
            x_values = np.linspace(min(random_x),max(random_x),4)[1:3]
            x_values = np.append(x_values,random_x)
            
            # finds f(x)
            y_values = list(func(x_values))
            # Gets indices of a sorted array
            indices = np.argsort(y_values)
            
            #picks the 4 smallest values to carry on with
            random_x = [x_values[i] for i in indices][0:4]
            random_y = [y_values[i] for i in indices][0:4]                        

            print("123")

           # random_x = [random.uniform(min(random_x),max(random_x)) for x in range(4)]
#            random_y = func(np.array(random_x))

        else:
            # Stores the next smallest value
            random_x.append(x_3)
            
            # Calculates the f(x) of four x values
            random_y = func(np.array(random_x))
        
    return

def sample_function(x):
    return x**3 + 3*x**2 #np.sin(1/5 * x)

if __name__ == "__main__":

    x_bot = -5
    x_top = 2
    sample_x = np.linspace(x_bot,x_top,100)

    plt.plot(sample_x,sample_function(sample_x))

    Parabolic_minimise(func = sample_function)

    plt.show()