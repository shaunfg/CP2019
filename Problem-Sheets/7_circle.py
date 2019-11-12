# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 13:29:15 2019

@author: sfg17
"""

import numpy as np
import random

def in_circle():
    return

accepted_x =[]
rejected_x = []

accepted_y =[]
rejected_y = []


for i in range(1000):
    x = random.uniform(0,1)
    y = random.uniform(0,1)
    
    radius = x**2 + y**2
    if radius <= 1/2:
        accepted_x.append(x)
        accepted_y.append(y)
    else:
        rejected_x.append(x)
        rejected_y.append(y)

fraction = len(accepted_x)/ len(rejected_x)

Area = fraction * (1*1)

circle_area = np.pi*1/4

difference = Area - circle_area

print(Area,circle_area)
print(difference)
