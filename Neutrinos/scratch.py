# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 16:10:28 2019

@author: sfg17
"""

def Error_Curvature(theta,unoscillated_rates,measured_events):
    
    t = theta 
    
    A = np.sin(1.267 * del_mass_square * L  / E) **2
    
    P = 1 - np.sin(2*t)**2 * A
    P_1 = - 4*np.sin(2*t)*np.cos(2*t) * A
    P_2 = - (8(np.cos(2*t)**2 - np.sin(2*t)**2) * A)
    
    
    l = P * unoscillated_rates
    l_1 = P_1 * unoscillated_rates
    l_2 = P_2 * unoscillated_rates
   
    NLL_2 = l_2 - measured_events * (l * l_2 - l_1 **2) / l**2
    
    return NLL_2

