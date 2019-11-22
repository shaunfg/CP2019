#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 19:10:25 2019

@author: ShaunGan
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 3.1 The data
def read_data(filename):
    # Reads data file into a dataframe
    data = pd.read_csv(filename)
    
    # Splits first 200 and last 200 data points
    df = data.iloc[:200].copy()
    df['unoscillated_flux'] = data.iloc[201:].copy().reset_index(drop = True)
    
    # Rename columns
    df.columns = ["oscillated_rate","unoscillated_rate"]
    
    # Append column for energy bins
    df["energy"] = np.arange(0.025,10.025,0.05)
    
    # Force float type for all data points
    df = df.astype(float)
    return df

def survival_probability(E,theta,del_mass_square,L):
    
    coeff = 1.267 * del_mass_square * L  / E
    P = 1 - np.sin(2*theta)**2 * np.sin(coeff) **2
    return P
    
def oscillated_prediction(thetas):
        # Calculate probabiliites of oscillation
    probs = [survival_probability(energies,thetas[i],del_m,L) for i in
                    range(len(thetas))]
    
    # Obtain unoscillated rates from data grame
    unosc_rates = data["unoscillated_rate"].tolist()
    
    # Convert to numpy arrays
    probs = np.array(probs)
    unosc_rates = np.array(unosc_rates)
    
    # Find oscillated rates
    osc_rates = probs * unosc_rates
    return osc_rates


def NLL(rates,N_observed_events):
    
    k = N_observed_events
    
    NLL_value = []
    
    for j in range(len(rates)):
        tmp = []
        l  = rates[j]

        for i in range(len(l)):
            
            if k[i] != 0:
                value = l[i] - k[i] + k[i] * np.log10(k[i]/l[i])
                tmp.append(value)
            else:   
                pass

        NLL_value.append(sum(tmp))
        
    return NLL_value
    
if __name__ == "__main__":    
    data = read_data("data.txt")
#    data = read_data("chris_data.txt")
    data["oscillated_rate"].hist(bins = 50)
    
    # Guess Parameters
    del_m = 2.9e-3 # adjusted to fit code data better
    L = 295
    
    # Energies and Theta values to vary
    energies = np.array(data['energy'].tolist())
    thetas = np.linspace(0,np.pi,200)
    oscillated = np.array(data["oscillated_rate"].tolist())
    unoscillated = np.array(data["unoscillated_rate"].tolist())
    predicted = oscillated_prediction([np.pi/4])
    
    # 
    plt.figure(figsize = (8,5))
    plt.bar(energies,oscillated,width = 0.05,alpha = 0.5)
    plt.xlabel("Energies/GeV")
    plt.ylabel("Rates")
    plt.title("Measured Data after oscillation")
    plt.plot(energies,predicted[0])



    plt.figure()
    plt.bar(energies,unoscillated,width = 0.05)
    plt.xlabel("Energies/GeV")
    plt.ylabel("Rates")
    plt.title("unoscillated")    

#    plt.figure()

    
    osc_rates = oscillated_prediction(thetas=thetas)
    
    NLL_array = NLL(osc_rates, oscillated)   
        
    plt.figure()
    plt.xlabel("Thetas")
    plt.ylabel("NLL")
    plt.plot(thetas, NLL_array)
    

    



    
    
