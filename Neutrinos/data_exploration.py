#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 19:10:25 2019

@author: ShaunGan
"""
import os
os.chdir("/Users/ShaunGan/Desktop/computational-physics/Neutrinos")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange,interp1d
from numpy.polynomial.polynomial import Polynomial
import random
import warnings

from plot_functions import plot_mass,plot_theta,plot_color_map,plot_steps
from plot_functions import plot_rates,plot_std

#%%
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

#%%
#data = read_data("//icnas3.cc.ic.ac.uk/sfg17/Desktop/Computational Physics/Neutrinos/data.txt")
data = read_data("data.txt")
#data = read_data("chris_data.txt")    
#data = read_data("matt_data.txt")
# Guess Parameters
del_m_square = 2.4e-3#4e-3 # adjusted to fit code data better
L = 295
t_IC = np.pi/4

# Prepare lists to be used in calculations
energies = np.array(data['energy'].tolist())
thetas = np.arange(0,np.pi,0.002)
oscillated = np.array(data["oscillated_rate"].tolist()) # measured
unoscillated = np.array(data["unoscillated_rate"].tolist()) # simulated

# Define tuple for more compact formatting
ICs = (L,energies)

#%% One Dimensional Minimisation (Section 3)
def survival_probability(E,theta,del_mass_square,L):
    
    coeff = 1.267 * del_mass_square * L  / E
    P = 1 - np.sin(2*theta)**2 * np.sin(coeff) **2
    return P

#%%
def cross_section(E,a,C):
            return C + a * E    

def oscillated_prediction(thetas,masses, cross_a = 0,C = 0):
    # Calculate probabiliites of oscillation
    
    if np.array(masses).size > np.array(thetas).size:
        if isinstance(masses, (np.ndarray,list)) == True:
            probs = [survival_probability(energies, thetas, masses[i], L) for i in
                     range(len(masses))]
        else:
            raise ValueError("Unknown type")
            
    elif np.array(masses).size < np.array(thetas).size:
        if isinstance(thetas,(np.ndarray,list)) == True:
            probs = [survival_probability(energies,thetas[i],masses,L) for i in
                     range(len(thetas))]
        else:
            raise ValueError("Unknown type")#     
    else:
        probs = [survival_probability(energies,thetas,masses,L)]       
#    
    # Obtain unoscillated rates from data grame
    unosc_rates = data["unoscillated_rate"].tolist()
    
    # Convert to numpy arrays
    probs = np.array(probs)
    unosc_rates = np.array(unosc_rates)
    
    # Calculate cross sections
    if cross_a == 0:
        cross_sections = 1 #Returns it to the original parameters
    else:
        cross_sections = cross_section(energies,cross_a,C)

    # Find oscillated rates
    osc_rates = probs * unosc_rates * cross_sections
    return osc_rates

#%%
del_m_square = 2.9e-3#4e-3 # adjusted to fit code data better

predicted = oscillated_prediction(t_IC,del_m_square)
plot_rates(energies,unoscillated,oscillated,predicted,[t_IC,del_m_square,0,0])

#%%
def NLL(theta_values,del_m,cross_a = 0,cross_C = 0):
    rates = oscillated_prediction(thetas=theta_values,masses=del_m,
                                  cross_a=cross_a,C = cross_C)
    k = oscillated
    
    NLL_value = []
    for j in range(len(rates)):
        tmp = []
        l  = rates[j]

        for i in range(len(l)):
            
            if k[i] != 0:
                value = l[i] - k[i] + k[i] * np.log(k[i]/l[i])
                tmp.append(value)
            else:   
                pass

        NLL_value.append(sum(tmp))
    if len(NLL_value) ==1:
        return NLL_value[0]
    else:
        return NLL_value

#%% Calculate NLL 
NLL_thetas = NLL(thetas,del_m_square)

plot_mass(func = NLL,theta = t_IC)
plot_theta(func = NLL)

#%%
def Parabolic(guess_x,IC = None,func = None,param = 'theta',limit= np.pi/4,
              return_smallest = False):
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
            # if denomiator is zero, pick mid point aaaof x
            x_list = [x for x in x_list if x <= x_list[np.argmax(x_list)]]
            x_list = [x for x in x_list if x >= x_list[np.argmin(x_list)]]
            x_3 = x_list[0]
        return x_3
    
#    IC = np.array(IC)

    if param == "theta":
        NLL_func = lambda k: func(np.array(k),IC)
    elif param == "mass":
        NLL_func = lambda k: func(IC,np.array(k))
    elif param == "1D_general":
        y = 0
        NLL_func = lambda k: func(k,y)
    else:        
        raise ValueError("Invalid param type")
    vals_x = guess_x#[random.uniform(x_bottom,x_top) for x in range(3)]
    vals_y = NLL_func(vals_x)

    x_3 = _find_next_point(vals_x,vals_y)
    x_3_last = x_3 + 10
    vals_x.append(x_3)
    vals_y = NLL_func(vals_x)

    end_count = 0
    while end_count<5:  
        vals_x.sort()        
        # Find maximum f(x) values
        max_idx = np.argmax(vals_y)
        
        # delete the smallest x value
        del vals_x[max_idx]
                
        # Finds the new f(x) values
        vals_y = NLL_func(vals_x)

        # Finds the next minimum value
        x_3_last = x_3
        x_3 = _find_next_point(vals_x, vals_y)

        # Check for mixture of negative and positive curvature
        if NLL_func(x_3) > all(vals_y):
            warnings.warn("Interval has positive & negative curvature", Warning) 
#            print(vals_x)
            vals_x.append(x_3)
            # finds 2 additional values from max and min of interval
            x_values = np.linspace(min(vals_x),max(vals_x),4)[1:3]
            x_values = np.append(x_values,vals_x)
            
            # finds f(x)
            y_values = list(NLL_func(x_values))
            
            # Gets indices of a sorted array
            indices = np.argsort(y_values)
            
            #picks the 4 smallest values to carry on with
            vals_x = [x_values[i] for i in indices][0:4]
            vals_y = [y_values[i] for i in indices][0:4]

        else:
            # Stores the next smallest value
            vals_x.append(x_3)
            
            # Calculates the f(x) of four x values
            vals_y = NLL_func(vals_x)
        
        if round(x_3,15) == round(x_3_last,15):
            end_count +=1
        else:
            end_count = 0
            
    if return_smallest == True:
        return x_3
    else:
        
        max_idx = np.argmax(vals_y)
        del vals_x[max_idx]      
        vals_y = NLL_func(vals_x)
    
        return vals_x,vals_y
#%%
# Parabolic Minimise
vals_x,vals_y = Parabolic(IC=del_m_square,guess_x = [0.2,0.4,0.6],func = NLL)

# Minimums based on the parabolic minimiser
min_theta_1D = min(vals_x)
min_NLL_1D = min(vals_y)    

#%% Finding Errors on minimum

def Error_abs_addition(variables,NLL_values,min_val_x,min_NLL,above_min):
    """
    Finds error by +/- 0.5, on the first minimum for a parabolic fit.
    
    Points are linearly interpolated. 
    
    min_val_x,min_NLL from parabolic minmised fit.
    """
    # Form pandas datraframe
    comparison_df = pd.DataFrame({"Var":variables,"NLL":NLL_values})
    # Localise to first minimum    
    comparison_df = comparison_df.loc[(comparison_df.Var < np.pi/4)]
    #TODO: do minus
    
    comparison_df = comparison_df.loc[(comparison_df.NLL < min_NLL + 2*above_min) & (comparison_df.Var < 1.5)]
    
#    print(comparison_df)
    
    # Finds LHS and RHS of the minimum
    LHS = comparison_df.loc[comparison_df.Var < min_val_x]
    RHS = comparison_df.loc[comparison_df.Var > min_val_x]
    
#    print(min_val_x)
    
    # Interpolate to find continous value
    func_LHS = interp1d(LHS["NLL"],LHS["Var"]) # Flipped to inverse function
    func_RHS = interp1d(RHS["NLL"],RHS["Var"]) # Flipped to inverse function

    #Takes maximum value
    values = np.array([func_LHS(min_NLL + 0.5)-min_val_x,func_RHS(min_NLL + 0.5)-min_val_x])
    std_dev = values
        
    return std_dev

# Calculate error form +/- 0.5 through linear interpolation. 
std_t_abs = Error_abs_addition(thetas,NLL_thetas,min_theta_1D,min_NLL_1D,0.7)

#%%
def Error_Curvature(unoscillated_rates,measured_events,parabolic_x,parabolic_y):    
    # Calculate second derivative of Lagrange polynomial fit
    x = parabolic_x
    y = parabolic_y
    
    # Based on the parabolic minimiser
    poly = lagrange(x,y)
    coefficients = Polynomial(poly).coef
    
    # Just consider curvature, ax**2 = 0.5
    std_theta = np.sqrt(0.5/coefficients[0])
    
    return std_theta

# Calculate error from +/- 0.5 through fitting to second derivative  
std_t_curv = Error_Curvature(unoscillated,oscillated,vals_x,vals_y)


#%%
def Error_2nd_Dev(unoscillated_rates,measured_events,del_mass_square,IC,parabolic_x,parabolic_y):
    """
    Calculates error by finding difference in second derivatives. 
    
    Params:
        IC = tuple of del_mass_square,L,energies
    """
    # Find minimum theta and minimum NLL
    min_val = min(parabolic_x)
    
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
    NLL_2 = l_2 - measured_events * (l * l_2 - l_1 **2) / l**2 
    
    # Curvature is equal to NLL_2 = 2a
    curvature_original = sum(NLL_2) /2
    
    # Calculate second derivative of Lagrange polynomial fit
    x = parabolic_x# 
    y = parabolic_y
    
    # Based on the parabolic minimiser
    poly = lagrange(x,y)
    coefficients = Polynomial(poly).coef
    
    # Find difference in curvature value
    curvature_del = curvature_original - coefficients[0]
    
    # Just consider curvature, ax**2 = 0.5
    std_theta = np.sqrt(0.5/curvature_del)
    
    return std_theta

std_t_dev = Error_2nd_Dev(unoscillated,oscillated,del_m_square,ICs,vals_x,vals_y)

print("error 2nd dev", std_t_dev)
#%%
plot_std(min_theta_1D,del_m_square,std_t_abs,std_t_curv,std_t_dev,NLL,thetas,NLL_thetas,min_NLL_1D)

print("===========================")
print("From 1D Parabolic Minimiser")
print("===========================")
print("Absolute Error: Min Theta = {:.6f} +/- {:.6f}".format(min_theta_1D,max(abs(std_t_abs))))
print("Curvature Error: Min Theta = {:.6f} +/- {:.6f}".format(min_theta_1D,std_t_curv))
print("2nd Dev Error: Min Theta = {:.6f} +/- {:.6f}".format(min_theta_1D,std_t_dev))

#%% Two Dimensional Minimisation (Section 4)
def Univariate(func,guess_a,guess_b):
    min_x_mass = guess_a
    min_x_thetas = guess_b
    
    # Find Radius
    R = np.sqrt(min(min_x_thetas)**2 + min(min_x_mass)**2)
    R_last = R + 10
    end_count = 0
    
    mass_step = [min(min_x_mass)]
    theta_step = [min(min_x_thetas)]
    
    while end_count <3:        
        min_x_mass,min_y_mass = Parabolic(func = func,IC=min(min_x_thetas), guess_x=min_x_mass,param = "mass")
        mass_step.append(min(min_x_mass))
        theta_step.append(min(min_x_thetas))
    
        min_x_thetas, min_y_t = Parabolic(func = func,IC=min(min_x_mass), guess_x=min_x_thetas)
        mass_step.append(min(min_x_mass))
        theta_step.append(min(min_x_thetas))
        
        # Finds if value repeats 3 times
        R = np.sqrt(min(min_x_thetas)**2 + min(min_x_mass)**2)                
        if round(R,18) == round(R_last,18):
            end_count +=1
        else:
            end_count = 0
        R_last = R
        
#        print(thetas)

    return [min_x_mass,min_y_mass,min_x_thetas,min_y_t, mass_step,theta_step]

#%%
# Obtain Univariate minimisation    
values = Univariate(func = NLL,guess_a = [1.5e-3,2e-3,3e-3],guess_b =[0.2, 0.5, 1])
min_masses_2D,min_NLL_masses,min_thetas_2D,min_NLL_thetas,mass_step,theta_step = tuple(values)

# Find Minimum values of masses and thetas
min_mass_2D = min(min_masses_2D)
min_theta_2D = min(min_thetas_2D)

# Find errors on univariate
std_mass_2D = Error_Curvature(unoscillated,oscillated,min_masses_2D,
                      min_NLL_masses)
std_t_2D = Error_Curvature(unoscillated,oscillated,min_thetas_2D,
                      min_NLL_thetas)
   
print("============================")
print("From 2D Univariate Minimiser")
print("============================") 
print("Minimum theta = {:.4f} +/- {:.4f}".format(min_theta_2D,std_t_2D))
print("Minimum mass = {:.7f} +/- {:.7f}".format(min_mass_2D,std_mass_2D))
print("Minimum NLL = {:.4f}".format(min(min_NLL_thetas)))
#%%
# Plots steps on colour map
plot_color_map(np.linspace(0,np.pi,100),np.linspace(0,5e-3,100),NLL)
plot_steps(theta_step,mass_step,"$\Theta$","$\Delta m^2$",
           "Steps taken for Univariate minimisation of NLL",font_size = "small")

plot_mass(func = NLL,theta = min(min_thetas_2D),val = round(min(min_thetas_2D),4))
plt.plot(min(min_masses_2D),NLL(min(min_thetas_2D),min(min_masses_2D)),'x')

plot_theta(func= NLL,mass = min(min_masses_2D),val = round(min(min_masses_2D),6))
plt.plot(min(min_thetas_2D),NLL(min(min_thetas_2D),min(min_masses_2D)),'x')

predicted = oscillated_prediction(min_theta_2D,min_mass_2D)
plot_rates(energies,unoscillated,oscillated,predicted,min_p = [min_theta_2D,min_mass_2D,0,0])

#%% Simulated Annealing 
def Simulated_Annealing(func,N,T_start,T_div,xy_step,guesses,limit = None,
                        PCS_mode = True,PE =0.3, PL = 0.29):
    
    def Thermal(E,T):
        k_b = 1# 1.38e-23
        return np.exp(- E / (k_b * T))
    
    def PCS(T_o,T_f,T_div,i):
        """
        probabilistic cooling scheme
        """
        N = abs(T_f - T_o) / T_div
        a = 0.01
        
        A = (T_o - T_f)*(N+1)/N
        B = T_o - A
        T_i = (A/(i+1)+B)*PE + (a*T_o/np.log(1+i))*PL
        return T_i

    t_values = []
    m_values = []
    c_values = []
    
    x = [0]* N
    x_dash = [0] * N
    h = xy_step# step sizes (initial conditions)
    
    # Set guesses 
    for i in range(len(x)):
        x[i] = random.uniform(guesses[i][0],guesses[i][1])
    
    success_count = 0
    count = 1
    T = T_start

    print("Starting Temperature {:.2f} K".format(T))
    while T > 1e-15:

        if count % 1000 == 0:
            print("Temperature {:.4f} K".format(T))
            t_values.append(x_dash[0])
            m_values.append(x_dash[1])
#            
c_values.append(x_dash[2])
            
        for i in range(len(x)):
            x_dash[i] = random.gauss(x[i],h[i])

#        if limit != None:
        while x_dash[0] < 0:
            x_dash[0] = random.uniform(x[0]-h[0],limit)

        del_f = np.array(func(*x_dash)) - np.array(func(*x)) 
        p_acc = Thermal(del_f,T)
       
        if del_f <= 0:# if next energy value is smaller, then update
            for i in range(len(x)):
                x[i] = x_dash[i]            
            success_count +=1
            t_values.append(x[0])
            m_values.append(x[1])
        
        elif p_acc > random.uniform(0,1):# probability of selecting other point
#            print(p_acc)
            for i in range(len(x)):
                x[i] = x_dash[i]            
            success_count +=1
            t_values.append(x[0])
            m_values.append(x[1])
            
        else:
            pass 
        
        if PCS_mode == False:
            T -= T_div
        else:
            T = PCS(T_start,0,T_div,count)
        count+=1
        
    t_values.append(x[0])
    m_values.append(x[1])
#    c_values.append(x_dash[2])

    NLL_value = func(*x) 
    print("Minimum function value = ",NLL_value)
    print("Minimum Values = ",x)

    print("Efficiency = {:.4f}%".format(success_count/count * 100))    
    print(count)      
    return x,NLL_value,[t_values,m_values,c_values]

#%% Sample Run - 2D Simulated Annealing with NLL 
min_p,min_NLL,steps = Simulated_Annealing(NLL,N = 2,T_start = 1000,T_div = 1,
                                      xy_step = [0.1,0.5e-3],
                                          guesses = [[0.7,0.8],[2e-3,3e-3]])

print("... Sample RUN")
print("===========================")
print("From 2D Simulated Annealing")
print("===========================") 
print("Minimum theta = {:.4f}".format(min_p[0]))
print("Minimum mass = {:.7f}".format(min_p[1]))
print("Minimum NLL = {:.4f}".format(min_NLL))
#%%
predicted = oscillated_prediction(*min_p)
plot_rates(energies,unoscillated,oscillated,predicted,val = (min_p[0],min_p[1]))

#%% Sample Run - 3D Simulated Annealing with Cross Section
#min_p,min_NLL,steps = Simulated_Annealing(NLL,N = 3,T_start = 1000,T_div = 1,
#                                          xy_step = [0.05,0.5e-3,0.1],
#                                          guesses = [[0.7,0.8],[2e-3,3e-3],
#                                                     [0.3,0.35]]) 
#
min_p,min_NLL,steps = Simulated_Annealing(NLL,N = 3,T_start = 1000,T_div = 1,
                                          xy_step = [0.15,0.5e-3,0.1],
                                          guesses = [[0.7,0.8],[2e-3,3e-3],
                                                     [1,2]])
#min_p,min_NLL,steps = Simulated_Annealing(NLL,N = 3,T_start = 1000,T_div = 1,
#                                          xy_step = [0.15,0.5e-3,0.1],
#                                          guesses = [[0.2,1.2],[1e-3,5e-3],
#                                                     [1,2]],PE = 0.3)
                
print("... Sample RUN")
print("===========================")
print("From 3D Simulated Annealing")
print("===========================") 
print("Minimum theta = {:.4f}".format(min_p[0]))
print("Minimum mass = {:.7f}".format(min_p[1]))
print("Minimum cross section = {:.4f}".format(min_p[2]))
print("Minimum NLL = {:.4f}".format(min_NLL))

#%%


# Plot energies against rates 
#min_p.append(0.4)
min_p.append(0)
predicted = oscillated_prediction(*min_p)
plot_rates(energies,unoscillated,oscillated,predicted,min_p)
         

#%% Plot theta 
labels = [round(x,4) for x in min_p]
plot_theta(NLL,min_p[1],min_p[2],val= tuple(labels))
plt.plot(min_p[0], NLL(*min_p),'x')


#%% Plot steps for simulated Annealing
plot_color_map(np.linspace(0,np.pi,100),np.linspace(0,5e-3,100),NLL)
plot_steps(steps[0],steps[1],"$ \Theta_{23}$","$\Delta m_{23}^2$","Simulated Annealing for NLL")
plt.legend(loc=1, prop={'size': 12})

#%% 4D minimisation, with offset
min_p,min_NLL,steps = Simulated_Annealing(NLL,N = 4,T_start = 1000,T_div = 1,
                                          xy_step = [0.15,1e-3,0.2,0.05],
                                          guesses = [[0.7,0.8],[2e-3,3e-3],
                                                     [0.7,0.8],[0.4,0.5]])
#%%
predicted = oscillated_prediction(*min_p,)
plot_rates(energies,unoscillated,oscillated,predicted,min_p )
                         
#%%
predicted = oscillated_prediction(min_p[0],min_p[1],min_p[2],0.4)
labels = [round(x,4) for x in min_p]
labels[3] = 0.4
plot_rates(energies,unoscillated,oscillated,predicted,min_p = tuple(labels))
         
#%% Calculate Errors for 2D Simulated Annealing Parameters
thetas = []
masses = []
NLL_l = []

for i in range(10):
    min_p,min_NLL,steps = Simulated_Annealing(NLL,N = 2,T_start = 1000,T_div = 1,
                                              xy_step = [0.1,0.5e-3],
                                              guesses = [[0.7,0.8],[2e-3,3e-3]])
    thetas.append(min_p[0])
    masses.append(min_p[1])
    NLL_l.append(min_NLL)


print("===========================")
print("From 2D Simulated Annealing")
print("===========================") 
print("Minimum theta = {:.4f} +/- {:.4f}".format(np.mean(thetas),np.std(thetas)))
print("Minimum mass = {:.7f} +/- {:.7f}".format(np.mean(masses),np.std(masses)))
print("Minimum NLL = {:.4f} +/- {:.4f}".format(np.mean(NLL_l),np.std(NLL_l)))

#%% Calculate Errors for 3D Simulated Annealing Parameters
thetas = []
masses = []
cross = []
NLL_l = []

for i in range(10):#[0.05,0.1,0.15,0.2,0.25,0.3]:
    min_p,min_NLL,steps = Simulated_Annealing(NLL,N = 3,T_start = 1000,T_div = 1,
                                          xy_step = [0.15,0.5e-3,0.1],
                                          guesses = [[0.7,0.8],[2e-3,3e-3],
                                                     [1,2]])
    thetas.append(min_p[0])
    masses.append(min_p[1])
    cross.append(min_p[2])
    NLL_l.append(min_NLL)
    
thetas = [[abs(x-np.pi/2) for x in thetas if x > np.pi/4], 
        [x for x in thetas if x < np.pi/4]]
thetas = [item for sublist in abc for item in sublist]    

print("===========================")
print("From 3D Simulated Annealing")
print("===========================") 
print("Minimum theta = {:.4f} +/- {:.4f}".format(np.mean(thetas),np.std(thetas)))
print("Minimum mass = {:.7f} +/- {:.7f}".format(np.mean(masses),np.std(masses)))
print("Minimum cross section = {:.4f} +/- {:.4f}".format(np.mean(cross),np.std(cross)))
print("Minimum NLL = {:.4f} +/- {:.4f}".format(np.mean(NLL_l),np.std(NLL_l)))

#%% Test Function: Ackley
def Ackley(x,y):
    x = np.array(x)
    y = np.array(y)
    A = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))
    B = - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
    return A + B + np.exp(1) + 20

sample_x = np.linspace(-5,5,1000)
plt.plot(sample_x, Ackley(sample_x,0))

# Parabolic minimisaition
test_x, test_y = Parabolic(guess_x = [-5,0,5],func = Ackley,param = "1D_general")
print("1D Parabolic Min for Ackley x = {} where (y = 0)".format(min(test_x)))

# Univariate minimisation
test = Univariate(func = Ackley,guess_a = [-0.5,0,0.5],guess_b = [-0.5,0,0.5])
print("2D Univariate Min for Ackley: x = {},y = {}".format(min(test[0]),min(test[2])))

#%% Simulated Annealing test with Ackley function 
min_p,min_NLL,steps = Simulated_Annealing(Ackley,N = 2,T_start = 100,T_div =0.01,xy_step = [0.2,0.2],
                      guesses = [[-5,5],[-5,5]],PE = 0.8,PL = 0.1)

plot_color_map(np.linspace(-5,5,100),np.linspace(-5,5,100),Ackley)
plot_steps(steps[0],steps[1],"x","y","Simulated Annealing for Ackley")
plt.legend(loc=1, prop={'size': 12})

#%% Test Function: sphere
def sphere(x,y):
    x = np.array(x)
    y = np.array(y)
    return (x-1)**2 + (y)**2

test_x = Parabolic(guess_x = [-5,0,5],func = sphere,param = "1D_general",
                           return_smallest = True)
print("1D Parabolic Min for 2D sphere x = {} where (y = 0)\n".format(test_x))

test = Univariate(func = sphere,guess_a = [-0.5,0,0.5],guess_b = [-0.5,0,0.5])
print("2D Univariate Min for sphere: x = {},y = {}".format(min(test[2]),min(test[1])))


#%%
#TODO: comment why do m first -- plot graphs
"""
m very bumpy, so better to find the minimum of m first and then theta, so that
the minimisation doesn't get stuck in a local minima in m. 
"""
#TODO: Comment on merit of both methods of error
"""
Error in curvature is good because it follows from a parabolic fit, so would
    give a more accurate value following the minimisation

Error in +/- 0.5 is also good because it gives roughly the same value, but then
    is faster
"""
    #TODO: Comment on annealing better than univariate?
"""
Annealing doesn't get stuck in local minima. but univariate is more direct. 
"""
    #TODO: Comment on cross section?
"""
With adding cross section, minimises to pi/4
"""
    #TODO: comment Error as pi/4?
"""
expect the error to reduce?
"""
