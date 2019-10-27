#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 17:57:22 2019

@author: ShaunGan
"""
#TODO: Sampling, Aliasing, Padding

import numpy as np
import matplotlib.pyplot as plt

def g(t):
    return 1/np.sqrt(2*np.pi) * np.exp(-t**2/2)

def h(t):
    
    values = []
    
    for i in range(len(t)):
        if t[i]>=3 and t[i]<=5:
            h = 4
        else:
            h = 0
        values.append(h)
        
    return values

def Convolution(time_values,func_1,func_2):
    """
    F(f*g) = F(f)F(g)
    """
    
    fft_first = np.fft.fft(func_1(time_values))    
    fft_second = np.fft.fft(func_2(time_values))
    
    fft_convoluted = fft_first * fft_second
#    print(np.real(fft_convoluted))
    
    convoluted = np.fft.ifft(fft_convoluted)
    print(convoluted)
    plt.figure()
    plot(func_1)
    plot(func_2)
    plt.figure()
#    plt.plot(time_values,fft_first,label = "F(h)")
#    plt.plot(time_values,fft_second,label = "F(g)")
    plt.plot(time_values,fft_convoluted,label = "F(h*g)")
    plt.legend()
    plt.figure()
    plt.plot(time_values,np.real(convoluted))

def plot(func):
    x = np.linspace(-10,10,1000)
    plt.plot(x,func(x))

# Padding to increase speed of fft
N_samples = 2**(10)

time = np.linspace(0,10,N_samples)

Convolution(time,h,g)

plt.show()

# plt.plot(time,np.sin(time))
# plt.figure()
# plt.plot(time,np.fft.fft(np.sin(time)))