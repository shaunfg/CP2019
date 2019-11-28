#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 19:10:25 2019

@author: ShaunGan
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.interpolate import lagrange
from scipy.interpolate import interp1d
from numpy.polynomial.polynomial import Polynomial


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
#%% One Dimensional Minimisation (Section 3)
def survival_probability(E,theta,del_mass_square,L):
    
    coeff = 1.267 * del_mass_square * L  / E
    P = 1 - np.sin(2*theta)**2 * np.sin(coeff) **2
    return P
    
def oscillated_prediction(thetas,del_m):
    # Calculate probabiliites of oscillation
    
    if isinstance(thetas,np.ndarray) == True:
        probs = [survival_probability(energies,thetas[i],del_m,L) for i in
                 range(len(thetas))]
    elif isinstance(thetas,list) == True:
        probs = [survival_probability(energies,thetas[i],del_m,L) for i in
                 range(len(thetas))]
    else:
        probs = survival_probability(energies,thetas,del_m,L)
    
    # Obtain unoscillated rates from data grame
    unosc_rates = data["unoscillated_rate"].tolist()
    
    # Convert to numpy arrays
    probs = np.array(probs)
    unosc_rates = np.array(unosc_rates)
    
    # Find oscillated rates
    osc_rates = probs * unosc_rates
    
    if isinstance(osc_rates,np.ndarray):
        return osc_rates
    else:
        print(osc_rates)
        return [osc_rates]

def NLL(theta_values,data,del_m):
#    print(theta_values)
    rates = oscillated_prediction(thetas=theta_values,del_m=del_m)
    k = data
    
    NLL_value = []
    
    for j in range(len(rates)):
        tmp = []
        l  = rates[j]

        for i in range(len(l)):
            
#            print(k)
            if k[i] != 0:
                value = l[i] - k[i] + k[i] * np.log10(k[i]/l[i])
                tmp.append(value)
            else:   
                pass

        NLL_value.append(sum(tmp))
        
    return NLL_value

def Parabolic_theta(measured_data,del_m= 2.915e-3,guess_x = [-2.2,0.1,2]):
    """
    generate f(x) from a set of x values,append the new x_3 value
    
    Parabolic minimiser is based on the intergral of the lagrange polynomial. 
    
    find f(x) with all four values
    save the smallest three values
    """

    def _find_next_point(x_list, y_list):
        """
        accepts list of x and y, each of 3 elements
        """
        x_0, x_1, x_2 = tuple(x_list)
        y_0, y_1, y_2 = tuple(y_list)
        num = (x_2 ** 2 - x_1 ** 2) * y_0 + (x_0 ** 2 - x_2 ** 2) * y_1 + (x_1 ** 2 - x_0 ** 2) * y_2
        denom = (x_2 - x_1) * y_0 + (x_0 - x_2) * y_1 + (x_1 - x_0) * y_2
        if denom != 0:
            x_3 = 1 / 2 * num / denom
        else:
            # if denomiator is zero, pick mid point of x
            x_list = [x for x in x_list if x <= x_list[np.argmax(x_list)]]
            x_list = [x for x in x_list if x >= x_list[np.argmin(x_list)]]
            x_3 = x_list[0]
        return x_3

    random_x = guess_x#[random.uniform(x_bottom,x_top) for x in range(3)]
    random_y = NLL(np.array(random_x),measured_data,del_m)

#    print(random_x)
    x_3 = _find_next_point(random_x,random_y)
    x_3_last = x_3 +10
    random_x.append(x_3)
    random_y = NLL(np.array(random_x),measured_data,del_m)

    while abs(x_3-x_3_last)>1e-10:  
        # Find maximum f(x) values
        max_idx = np.argmax(random_y)
        
        # delete the smallest x value
        del random_x[max_idx]
#        print("-",random_y)
                
        # Finds the new f(x) values
        random_y = NLL(np.array(random_x),measured_data,del_m)

        # Finds the next minimum value
        x_3_last = x_3
        x_3 = _find_next_point(random_x, random_y)
        
        # Check for negative curvature
        if NLL([x_3],measured_data,del_m)[0] > all(random_y):
            warnings.warn("Interval has positive & negative curvature", Warning) 
            
            random_x.append(x_3)

            # finds 2 additional values from max and min of interval
            x_values = np.linspace(min(random_x),max(random_x),4)[1:3]
            x_values = np.linspace(min(random_x),max(random_x),4)[1:3]
            x_values = np.append(x_values,random_x)
            
            # finds f(x)
            y_values = list(NLL(x_values,measured_data,del_m))
            # Gets indices of a sorted array
            indices = np.argsort(y_values)
            
            #picks the 4 smallest values to carry on with
            random_x = [x_values[i] for i in indices][0:4]
            random_y = [y_values[i] for i in indices][0:4]

        else:
            # Stores the next smallest value
            random_x.append(x_3)
            # Calculates the f(x) of four x values
            print(measured_data)
            random_y = NLL(np.array(random_x),measured_data,del_m)
    
    
    max_idx = np.argmax(random_y)
    del random_x[max_idx]        
    random_y = NLL(np.array(random_x),measured_data,del_m)
    random_x = random_x
    
#    min_y_value = NLL([x_3],measured_data)[0]

    return random_x,random_y

#%% Finding Errors on minimum

def Error_abs_addition(variables,NLL_values, min_val_x,min_NLL,above_min):
    """
    Finds error by +/- 0.5, on the first minimum for a parabolic fit.
    
    Points are linearly interpolated. 
    
    min_val_x,min_NLL from parabolic minmised fit.
    """
    # Form pandas datraframe
    comparison_df = pd.DataFrame({"Var":variables,"NLL":NLL_values})
    # Localise to first minimum
    comparison_df = comparison_df.loc[(comparison_df.NLL < min_NLL + 2*above_min) & (comparison_df.Var < 1.5)]
#    print(comparison_df)
    
    # Finds LHS and RHS of the minimum
    LHS = comparison_df.loc[comparison_df.Var < min_val_x]
    RHS = comparison_df.loc[comparison_df.Var > min_val_x]
    
    # Interpolate to find continous value
    func_LHS = interp1d(LHS["NLL"],LHS["Var"]) # Flipped to inverse function
    func_RHS = interp1d(RHS["NLL"],RHS["Var"]) # Flipped to inverse function

    #Takes maximum value
    values = np.array([func_LHS(min_NLL + 0.5)-min_val_x,func_RHS(min_NLL + 0.5)-min_val_x])
    std_dev = max(abs(values))
        
    return std_dev

def Error_Curvature(unoscillated_rates,measured_events,del_mass_square,IC,parabolic_x,parabolic_y):
    """
    Calculates error by finding difference in second derivatives. 
    
    Params:
        IC = tuple of del_mass_square,L,energies
    """
    
    # Find minimum theta and minimum NLL
    min_val = min(parabolic_x)
    min_NLL = min(parabolic_y)
    
    L,E = IC
    
    # Calculate second derivative of theoretical NLL value
    t = min_val
    A = np.sin(1.267 * del_mass_square * L  / E) **2
    
    # Calculate probabilities    
    P = 1 - np.sin(2*t)**2 * A
    P_1 = - 4*np.sin(2*t)*np.cos(2*t) * A
    P_2 = - (8* (np.cos(2*t)**2 - np.sin(2*t)**2) * A)
    
    # Calculate rates 
    l = P * unoscillated_rates
    l_1 = P_1 * unoscillated_rates
    l_2 = P_2 * unoscillated_rates
   
    # Calculate second dertivative
    NLL_2 = l_2 - measured_events * (l * l_2 - l_1 **2) / l**2 * 1/ np.log(10)
    
    # Curvature is equal to NLL_2 = 2a
    curvature_original = sum(NLL_2) /2
    
    # Calculate second derivative of Lagrange polynomial fit
    x = parabolic_x# 
    y = parabolic_y
    
    # Based on the parabolic minimiser
    poly = lagrange(x,y)
    coefficients = Polynomial(poly).coef
    
    # Shift polynomial by 0.5 and solve for roots
    coefficients[-1] = coefficients[-1] - min_NLL - 0.5
    roots = np.roots(coefficients)
    
    # Curvature = a
    curvature_fit = roots[0]
    
    # Find difference in curvature value
    curvature_del = curvature_original - curvature_fit
    
    # Just consider curvature, ax**2 = 0.5
    std_theta = np.sqrt(0.5/curvature_del)
    
    return std_theta

#%% Two Dimensional Minimisation (Section 4)

def oscillated_prediction_two(theta,masses):
    # Calculate probabiliites of oscillation

    if isinstance(masses, np.ndarray) == True:
        probs = [survival_probability(energies, theta, masses[i], L) for i in
                 range(len(masses))]
    elif isinstance(masses, list) == True:
        probs = [survival_probability(energies, theta, masses[i], L) for i in
                 range(len(masses))]
    else:
        probs = survival_probability(energies, theta, masses, L)

    # Obtain unoscillated rates from data grame
    unosc_rates = data["unoscillated_rate"].tolist()

    # Convert to numpy arrays
    probs = np.array(probs)
    unosc_rates = np.array(unosc_rates)

    # Find oscillated rates
    osc_rates = probs * unosc_rates

    if isinstance(osc_rates, np.ndarray):
        return osc_rates
    else:
        print(osc_rates)
        return [osc_rates]

def NLL_two(theta_value,mass_values, data):

    rates = oscillated_prediction_two(theta=theta_value,masses=mass_values)
    k = data

    NLL_value = []

    for j in range(len(rates)):
        tmp = []
        l = rates[j]

        for i in range(len(l)):

            #            print(k)
            if k[i] != 0:
                value = l[i] - k[i] + k[i] * np.log10(k[i] / l[i])
                tmp.append(value)
            else:
                pass

        NLL_value.append(sum(tmp))

    return NLL_value

def Parabolic_mass(measured_data,theta_IC, guess_m):
    """
    generate f(x) from a set of x values,append the new x_3 value

    Parabolic minimiser is based on the intergral of the lagrange polynomial.

    find f(x) with all four values
    save the smallest three values
    """

    def _find_next_point(x_list, y_list):
        """
        accepts list of x and y, each of 3 elements
        """
        x_0, x_1, x_2 = tuple(x_list)
        y_0, y_1, y_2 = tuple(y_list)
        num = (x_2 ** 2 - x_1 ** 2) * y_0 + (x_0 ** 2 - x_2 ** 2) * y_1 + (x_1 ** 2 - x_0 ** 2) * y_2
        denom = (x_2 - x_1) * y_0 + (x_0 - x_2) * y_1 + (x_1 - x_0) * y_2
        if denom != 0:
            x_3 = 1 / 2 * num / denom
        else:
            # if denomiator is zero, pick mid point of x
            x_list = [x for x in x_list if x <= x_list[np.argmax(x_list)]]
            x_list = [x for x in x_list if x >= x_list[np.argmin(x_list)]]
            x_3 = x_list[0]
        return x_3


    random_x = guess_m  # [random.uniform(x_bottom,x_top) for x in range(3)]
    random_y = NLL_two(theta_IC,random_x, measured_data)

    #    print(random_x)
    x_3 = _find_next_point(random_x, random_y)
    x_3_last = x_3 + 10
    random_x.append(x_3)
    random_y = NLL_two(theta_IC,np.array(random_x), measured_data)

    # while abs(x_3 - x_3_last) > 1e-10:
    # while abs(x_3_last - x_3) > 1e-
    for i in range(20):
        # Find maximum f(x) values & delete smallest x value
        max_idx = np.argmax(random_y)
        del random_x[max_idx]

        # Finds the new f(x) values
        random_y = NLL_two(theta_IC,random_x, measured_data)

        # Finds the next minimum value
        x_3_last = x_3
        x_3 = _find_next_point(random_x, random_y)

        # Check for negative curvature
        if NLL_two(theta_IC,[x_3], measured_data)[0] > all(random_y):
            # warnings.warn("Interval has positive & negative curvature", Warning)
            random_x.append(x_3)

            # finds 2 additional values from max and min of interval
            x_values = np.linspace(min(random_x), max(random_x), 4)[1:3]
            x_values = np.append(x_values, random_x)

            # finds f(x)
            y_values = list(NLL_two(theta_IC,x_values, measured_data))
            # Gets indices of a sorted array
            indices = np.argsort(y_values)

            # picks the 4 smallest values to carry on with
            random_x = [x_values[i] for i in indices][0:4]
            random_y = [y_values[i] for i in indices][0:4]

        else:
            # Stores the next smallest value
            random_x.append(x_3)
            # Calculates the f(x) of four x values
            print(measured_data)
            random_y = NLL_two(theta_IC,random_x, measured_data)
        # print(random_x)

    max_idx = np.argmax(random_y)
    del random_x[max_idx]
    random_y = NLL_two(theta_IC,random_x, measured_data)
    random_x = random_x

    return random_x, random_y

def Univariate(theta_IC, measured_data, guess_m,guess_x):
    min_x_mass,min_y_mass = Parabolic_mass(measured_data, theta_IC=theta_IC, guess_m=guess_m)
    min_x_thetas, min_y_t = Parabolic_theta(measured_data=measured_data, del_m=min(min_x_mass), guess_x=guess_x)
    return min_x_mass,min_y_mass,min_x_thetas,min_y_t

#%%
if __name__ == "__main__":
    data = read_data("data.txt")
#    data = read_data("chris_data.txt")
    
    # Guess Parameters
    del_m_square = 2.915e-3 # adjusted to fit code data better
    L = 295
    
    # Prepare lists to be used in calculations
    energies = np.array(data['energy'].tolist())
    thetas = np.arange(0,np.pi,0.002)
    oscillated = np.array(data["oscillated_rate"].tolist()) # measured
    unoscillated = np.array(data["unoscillated_rate"].tolist()) # simulated
    predicted = oscillated_prediction([np.pi/4],del_m_square)
    ICs = (L,energies)
    
    # Calculate NLL 
    NLL_thetas = NLL(thetas,oscillated,del_m_square)
    
    # Parabolic Minimise
    vals_x,vals_y = Parabolic_theta(oscillated,guess_x = [0.2,0.5,1],del_m=del_m_square)
    
    # Minimums based on the parabolic minimiser
    min_theta_1D = min(vals_x)
    min_NLL_1D = min(vals_y)
    print(min_theta_1D,"Check")
    
    # Calculate both errors, from +/- 0.5 and from difference in curvature
    std_t_abs = Error_abs_addition(thetas,NLL_thetas,min_theta_1D,min_NLL_1D,0.7)
    std_t_curv = Error_Curvature(unoscillated,oscillated,del_m_square,ICs,vals_x,vals_y)

    print(std_t_abs, std_t_curv)

    # Two dimensional minimisation
    theta = np.array([np.pi / 4])
    masses = np.linspace(1e-3,3e-3,1000)
    NLL_masses = NLL_two(theta,masses,oscillated)

    a,b,c,d = Univariate(theta,oscillated,[1e-3,2e-3,3e-3],guess_x =[0.2, 0.5, 1])
    min_masses_2D,min_NLL_masses,min_thetas_2D,min_NLL_thetas = (a,b,c,d)
    
    min_mass_2D = min(min_masses_2D)
    min_theta_2D = min(min_thetas_2D)
    abc = Error_abs_addition(masses,NLL_masses,min_mass_2D,min(min_NLL_masses),0.7)
    
    
    #(unoscillated,oscillated,min_mass_2D,ICs,min_masses_2D,min_NLL_masses)
    
#    abc = Error_Curvature(unoscillated,oscillated,min_mass_2D,ICs,min_masses_2D,min_NLL_masses)

    print(min_mass_2D,min_theta_2D)

    #TODO: OBTAIN ERROR
    #TODO: Set convergence limit
# ===================================PLOT======================================
    
#    # Plot hist
#    data["oscillated_rate"].hist(bins = 50)
    
#    # Measured oscillated data
#    plt.figure(figsize = (8,5))
#    plt.bar(energies,oscillated,width = 0.05,alpha = 0.5)
#    plt.xlabel("Energies/GeV")
#    plt.ylabel("Rates")
#    plt.title("Measured Data after oscillation")
#    plt.plot(energies,predicted[0])
#     
#    # Unoscillated simulated data 
#    plt.figure()
#    plt.bar(energies,unoscillated,width = 0.05)
#    plt.xlabel("Energies/GeV")
#    plt.ylabel("Rates")
#    plt.title("unoscillated")
#    
    # NLL against theta    
    plt.figure()
    plt.xlabel("Thetas")
    plt.ylabel("NLL")
    plt.plot(thetas, NLL_thetas)
#    plt.plot(samp_x,abc(samp_x))


    

    



    
    
