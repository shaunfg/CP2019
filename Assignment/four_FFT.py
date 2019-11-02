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
    """
    Gaussian Function, takes in an array of discrete data points. 
    
    Returns a normalized Gaussian
    """
    values = 1/np.sqrt(2*np.pi) * np.exp(-t**2/2)
    values_normalized = values/sum(values)
    return values_normalized

def h(t):
    """
    Tophat function, takes in an array of discrete data points
    """
    values = []
    for i in range(len(t)):
        if t[i]>=3 and t[i]<=5:
            h = 4
        else:
            h = 0
        values.append(h)
    return values

def Convolution(time_values,func_1,func_2,plot = True):
    """
    Performs a convolutions using the convolution theroem:
        F(f*g) = F(f)F(g)
    Thus finds convolutions of both functions separately, find the product, 
        and completes and inverse fourier transform
    
    The convoluted result is then required to undergo an fftshift, to shift
        the negative values to the beginning of the array. This is
        due to the nature of the algorithm
    
    -----------
    Parameters:
    -----------
    time_values: Array of discrete time values
    func1 : Tophat function
    func2 : Gaussian function
    plot : True/False, to plot the graphs. 
    """
    # Fourier transform of tophat and gaussian
    fft_first = np.fft.fft(func_1(time_values))
    fft_second = np.fft.fft(func_2(time_values))
    
    # Product of Fourier transforms
    fft_convoluted = fft_first * fft_second
    
    # Inverse Fourier Transform and shift for accurate plotting
    convoluted = np.fft.ifft(fft_convoluted)   
    convoluted = np.fft.fftshift(np.abs(convoluted))

    if plot == True:
        
        sample_x = np.linspace(-10,10,1000)
        
        # Shift fourier transforms of tophat and gaussian, for correct plotting
        fft_first_plot = np.fft.fftshift(fft_first)/np.sqrt(len(fft_first))
        fft_second_plot = np.abs(np.fft.fftshift(fft_second)/np.sqrt(len(fft_second)))
    
        # Plots tophat, gaussian and the convolution 
        fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize = (8,8))
        ax1.set_title("Fourier Transforms")
    
        ax1.plot(sample_x,func_1(sample_x),label = "Tophat")
        ax1.plot(time_values,fft_first_plot,label = "F(Tophat)")
        ax1.legend()
    
        ax2.plot(sample_x,func_2(sample_x),label = "Gaussian")
        ax2.plot(time_values,fft_second_plot,label = "F(Gaussian)")
        ax2.legend()
    
        ax3.plot(time_values,convoluted,label = "Convoluted Gauss*Tophat")
        ax3.plot(time_values,func_1(time_values),label = "Tophat")
        ax3.legend()

    else:
        pass
    
if __name__ == "__main__":
    
    # Padding to increase speed of fft
    N_samples = 2**(10)
    
    time = np.linspace(-10,10,N_samples)
    Convolution(time,h,g,plot = True)
    