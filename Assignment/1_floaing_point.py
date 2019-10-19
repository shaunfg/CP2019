# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 11:40:47 2019

@author: sfg17
"""

import numpy as np

def machineEpsilon(func=np.float64):
    machine_epsilon = func(1)
    while func(1)+func(machine_epsilon) != func(1):
        print(machine_epsilon)
        machine_epsilon_last = machine_epsilon
        machine_epsilon = func(machine_epsilon) / func(2)
    return machine_epsilon_last

a = machineEpsilon()
print(a)
print(np.finfo(float).eps)
