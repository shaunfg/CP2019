# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 13:29:15 2019

@author: sfg17
"""

import numpy as np
import random
import matplotlib.pyplot as plt
def in_circle():
    return

accepted_x =[]
rejected_x = []

accepted_y =[]
rejected_y = []
    
for i in range(10000):
    
    x = random.uniform(0,1)
    y = random.uniform(0,1)

    radius = (x-1/2)**2 + (y-1/2)**2
    if radius <= (1/2)**2:
        accepted_x.append(x)
        accepted_y.append(y)
    else:
        rejected_x.append(x)
        rejected_y.append(y)

N = (len(rejected_x)+len(accepted_x))
circle_area = np.pi*1/4

fraction = len(accepted_x)/ N
Area = fraction * (1*1)
difference = Area - circle_area

error = np.sqrt(0.5*(1-0.5)/N)

print(Area,circle_area)
print(difference)

plt.plot(accepted_x,accepted_y,'x')
plt.plot(rejected_x,rejected_y,'x')

plt.show()
#%%
def Intergrate_N_Sphere(n_dimensions):
    n =n_dimensions
    
    accepted = []
    rejected = []
    
    for i in range(10000):
        
        params = [random.uniform(0,1) for x in range(n)]
        radius = sum((np.array(params)-1/2)**2)
        if radius <= (1/2)**2:
            accepted.append(radius)
        else:
            rejected.append(radius)

    N = (len(rejected)+len(accepted))
    
    fraction = len(accepted)/ N
    Area = fraction * (1*1)
    difference = Area - circle_area
    
    error = np.sqrt(0.5*(1-0.5)/N)
    
    print(Area)

    
Intergrate_N_Sphere(8)