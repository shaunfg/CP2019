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
    values = 1/np.sqrt(2*np.pi) * np.exp(-t**2/2)
    values = values/sum(values)
    return values

def h(t):
    
    values = []
    
    for i in range(len(t)):
        if t[i]>=0 and t[i]<=5:
            h = 4
        else:
            h = 0
        values.append(h)
        
        
    return values

def Convolution(time_values,func_1,func_2):
    """
    F(f*g) = F(f)F(g)
    
    func1 = tophat
    func2 = gaussian
    """
    
    fft_first = np.fft.fft(func_1(time_values))
#    fft_first = np.fft.fftshift(fft_first)/ np.sqrt(len(time_values))    
#    fft_first = fft_first/sum(fft_first)
    
    fft_second = np.fft.fft(func_2(time_values))
#    fft_second = np.fft.fftshift(np.abs(fft_second)) / np.sqrt(len(time_values))
    
#    np.fft.fftshift(np.abs(np.fft.fft(g(x))))
    
    fft_convoluted = fft_first * fft_second
    
#    fft_convoluted = np.fft.fftshift(np.abs(fft_convoluted))/ np.sqrt(len(time_values))
#    print(np.real(fft_convoluted))
    
    convoluted = np.fft.ifft(fft_convoluted)   
    convoluted = np.fft.fftshift(np.abs(convoluted))
    
    print(convoluted)
    plt.figure()
    plot(func_1)
    plt.plot(time_values,fft_first,label = "F(tophat)")
    plt.figure()
    plot(func_2)    
    plt.plot(time_values,fft_second,label = "F(gaussian)")
    plt.figure()
    plt.plot(time_values,fft_convoluted,label = "F(top*gauss)")
    plt.legend()
    
    plt.figure()
    plt.plot(time_values,convoluted)

def plot(func):
    x = np.linspace(-10,10,1000)
    plt.plot(x,func(x))

# Padding to increase speed of fft
N_samples = 2**(10)

time = np.linspace(-10,10,N_samples)

Convolution(time,h,g)

plt.show()

# plt.plot(time,np.sin(time))
# plt.figure()
# plt.plot(time,np.fft.fft(np.sin(time)))

#%%

import matplotlib.pyplot as plt
import numpy as np

N = 128
x = np.arange(-5, 5, 10./(2 * N))
#y = np.exp(-x * x)
y_fft = np.fft.fftshift(np.abs(np.fft.fft(g(x))))# / np.sqrt(len(y))
plt.plot(x,y)
plt.plot(x,y_fft)
plt.show()
