# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 14:33:56 2019

@author: sfg17
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

def func(x,y):
    return x**2 + y**2

x = np.linspace(-5,5,100)
y = np.linspace(-5,5,100)


def plot_color_map(x,y,func):
    Z = []
    for i in range(len(x)):
        Z.append([func(x[i],y[j]) for j in range(len(y))])
    
    #print(Z)
    fig, (ax0) = plt.subplots(1, 1)
    
    c = ax0.pcolor(Z,cmap = 'viridis')
    ax0.set_title('default: no edges')
    
    space = 5
    
    ticks = np.linspace(0,len(Z[0]),space)
    plt.setp(ax0, xticks=ticks, xticklabels=np.linspace(min(x),max(x),space),
             yticks=ticks, yticklabels = np.linspace(min(y),max(y),space))

    fig.tight_layout()
    
    fig.colorbar(c, ax=ax0)
    
plot_color_map(x,y,func=func)
#
#plt.figure()
#plt.plot(x,func(x,1))
#
#plt.xticks(np.arange(5), ('Tom', 'Dick', 'Harry', 'Sally', 'Sue'))