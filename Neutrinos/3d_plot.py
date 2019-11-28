# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 19:53:51 2019

@author: sfg17
"""

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

from data_exploration import NLL
from data_exploration import read_data

data = read_data("data.txt")
measured_data = data["oscillated_rate"].tolist()

fig = plt.figure()
ax = plt.axes(projection='3d')

NLL()

def f(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))

x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)

ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax.set_title('surface');ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z');