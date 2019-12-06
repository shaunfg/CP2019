#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 23:28:48 2019

@author: ShaunGan
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_mass(func,theta = np.pi/4):
    masses = np.linspace(0,5e-3,1000)
    NLL_masses = func(theta,masses)
    
    plt.figure()
    plt.plot(masses,NLL_masses)
    plt.xlabel("$\Delta m^2$")
    plt.ylabel("NLL")
    plt.title("NLL against delta mass squared")
#    plt.plot(0.0026026,NLL(np.pi / 4,0.0026026),'x')
    
def plot_theta(func,mass = 2.4e-3,cross_a = 0):
    
    thetas = np.arange(0,np.pi,0.002)
    # Calculate NLL 
    NLL_thetas = func(thetas,mass,cross_a)
    
    # NLL against theta    
    plt.figure()
    plt.xlabel("Thetas")
    plt.ylabel("NLL")
    plt.plot(thetas, NLL_thetas)

def plot_color_map(x,y,func):
    Z = []
    for i in range(len(x)):
        Z.append([func(x[j],y[i]) for j in range(len(y))])
    
    fig,axes = plt.subplots(1,1,figsize=(10,8))
    
    c = axes.imshow(Z,origin="lower",interpolation = "None",aspect = "auto",
                    extent = [min(x),max(x),min(y),max(y)],cmap = 'viridis')
    
    plt.xlabel("$\Theta$")
    plt.ylabel("$\Delta m^2$")
    plt.title("NLL colour map of $\Delta m^2$ and $\Theta$")
    
    fig.colorbar(c,ax=axes, label='NLL')
#    print("\n",Z)
#plot_color_map(np.linspace(0,np.pi/2,100),np.linspace(0,5e-3,100),NLL)    

def plot_steps(t_values,m_values,xlabel,ylabel,title):
#    plt.figure(figsize = (10,8))
    plt.plot(t_values,m_values,color = '#1f77b4',label = "Min-Trajectory")    
    plt.plot(t_values,m_values,'o',ms = 2,color = '#ff7f0e',label = "Step point")    
    plt.plot(t_values[-1],m_values[-1],'o', color = '#17becf',label = "Minimum point")  
             
    plt.xlabel(xlabel,fontsize = 18)
    plt.ylabel(ylabel,fontsize = 18)
    plt.title(title,fontsize= 18)
    
    return
