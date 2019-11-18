#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 19:10:25 2019

@author: ShaunGan
"""

from NLL import NLL

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
    df.columns = ["N_observed_events","unoscillated_rate"]
    
    # Append column for energy bins
    df["energy"] = np.arange(0.05,10.05,0.05)
    
    # Force float type for all data points
    df = df.astype(float)
    return df

def survival_probability(E):
    
    theta = np.pi/4
    del_mass_square = 2.4e-3
    L = 295
    
    coeff = 1.267 * del_mass_square * L  / E
    P = 1 - np.sin(2*theta)**2 * np.sin(coeff) **2
    return P
    
def calc_decay_rate(df):
    plt.figure()
    energies = np.array(df['energy'].tolist())
    df["osc_prob"] = survival_probability(energies)
    plt.plot(energies, df["osc_prob"])

    plt.figure()
    unosc_flux = np.array(df['unoscillated_rate'].tolist())
    df["osc_rate"] = df["unoscillated_rate"] * df["osc_prob"]
    plt.plot(energies,df["osc_rate"] )    
    
    return df
    
if __name__ == "__main__":
        
    data = read_data("data.txt")
    data["N_observed_events"].hist(bins = 50)
    
    data = calc_decay_rate(data)
    
    NLL_value = NLL(data["osc_rate"],data["N_observed_events"])
    
    
