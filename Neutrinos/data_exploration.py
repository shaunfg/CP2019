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
import random

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

data = read_data("//icnas3.cc.ic.ac.uk/sfg17/Desktop/Computational Physics/Neutrinos/data.txt")
    
#%%
# Guess Parameters
del_m_square = 2.4e-3 # adjusted to fit code data better
L = 295

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
def cross_section(E,a):
            return 1 + a * E    

def oscillated_prediction(thetas,masses, cross_a = 0):
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
        cross_sections = 1
    else:
        cross_sections = cross_section(energies,cross_a)

    # Find oscillated rates
    osc_rates = probs * unosc_rates * cross_sections
    return osc_rates

#%%
predicted = oscillated_prediction(np.pi/4,del_m_square)

fig,axes = plt.subplots(2,1,figsize = (9,10))

# Unoscillated simulated data 
axes[0].bar(energies,unoscillated,width = 0.05)
axes[0].set_xlabel("Energies/GeV")
axes[0].set_ylabel("Rates")
axes[0].set_title("Unoscillated Rates")

# Measured oscillated data
axes[1].bar(energies,oscillated,width = 0.05,alpha = 0.5,label = "Measured Oscillated Data")
axes[1].set_xlabel("Energies/GeV")
axes[1].set_ylabel("Rates")
axes[1].set_title("Measured Data after oscillation")
axes[1].plot(energies,predicted[0],color = '#ff7f0e',label = "Predicted Oscillation Data")
axes[1].legend()

#%%

def NLL(theta_values,del_m,cross_a = 0):
#    print(theta_values)
    rates = oscillated_prediction(thetas=theta_values,masses=del_m,cross_a=cross_a)
    k = oscillated
    
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
    if len(NLL_value) ==1:
        return NLL_value[0]
    else:
        return NLL_value

#%%
# Calculate NLL 
NLL_thetas = NLL(thetas,del_m_square)
def plot_theta(mass = del_m_square):
    
    thetas = np.arange(0,np.pi,0.002)
    # Calculate NLL 
    NLL_thetas = NLL(thetas,mass)
    
    # NLL against theta    
    plt.figure()
    plt.xlabel("Thetas")
    plt.ylabel("NLL")
    plt.plot(thetas, NLL_thetas)


#%%
def Parabolic(guess_x,IC = None,func = None,param = 'theta',limit= np.pi/4):
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

#    print(vals_x)
    x_3 = _find_next_point(vals_x,vals_y)
    x_3_last = x_3 +10
    vals_x.append(x_3)
    vals_y = NLL_func(vals_x)

    
    end_count = 0
    while end_count<2:  
        vals_x.sort()        
        # Find maximum f(x) values
        max_idx = np.argmax(vals_y)
        
        # delete the smallest x value
        del vals_x[max_idx]
#        print("-",vals_y)
                
        # Finds the new f(x) values
        vals_y = NLL_func(vals_x)

        # Finds the next minimum value
        x_3_last = x_3
        x_3 = _find_next_point(vals_x, vals_y)
        print(x_3)
        # Check for negative curvature
        if NLL_func(x_3) > all(vals_y):
            warnings.warn("Interval has positive & negative curvature", Warning) 
            
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
        
        if round(x_3,18) == round(x_3_last,18):
            end_count +=1
        else:
            end_count = 0

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

def Error_abs_addition(variables,NLL_values, min_val_x,min_NLL,above_min):
    """
    Finds error by +/- 0.5, on the first minimum for a parabolic fit.
    
    Points are linearly interpolated. 
    
    min_val_x,min_NLL from parabolic minmised fit.
    """
    # Form pandas datraframe
    comparison_df = pd.DataFrame({"Var":variables,"NLL":NLL_values})
    # Localise to first minimum
    
    comparison_df = comparison_df.loc[(comparison_df.Var < np.pi/4)]
    comparison_df = comparison_df.loc[(comparison_df.NLL < min_NLL + 2*above_min) & (comparison_df.Var < 1.5)]
    
    # Finds LHS and RHS of the minimum
    LHS = comparison_df.loc[comparison_df.Var < min_val_x]
    RHS = comparison_df.loc[comparison_df.Var > min_val_x]
    
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
    x = parabolic_x# 
    y = parabolic_y
    
    # Based on the parabolic minimiser
    poly = lagrange(x,y)
    coefficients = Polynomial(poly).coef
    
    # Just consider curvature, ax**2 = 0.5
    std_theta = np.sqrt(0.5/coefficients[0])
    
    return std_theta

# Calculate error from +/- 0.5 through fitting to second derivative  
std_t_curv = Error_Curvature(unoscillated,oscillated,vals_x,vals_y)

print("\nMin Theta 1D Parabolic Minimiser = {:.6f} +/- {:.6f} or {:.6f}".format(min_theta_1D,max(abs(std_t_abs)),std_t_curv))

#%%
def Error_2nd_Dev(unoscillated_rates,measured_events,del_mass_square,IC,parabolic_x,parabolic_y):
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
    
    # Find difference in curvature value
    curvature_del = curvature_original - coefficients[0]
    
    # Just consider curvature, ax**2 = 0.5
    std_theta = np.sqrt(0.5/curvature_del)
    
    return std_theta

std_t_dev = Error_2nd_Dev(unoscillated,oscillated,del_m_square,ICs,vals_x,vals_y)

print("error 2nd dev", std_t_dev)
#%%
points = []

points_abs = [min_theta_1D + x for x in std_t_abs]
NLL_points_abs = NLL(np.array(points_abs),del_m_square)

points_curv = [min_theta_1D + x for x in [std_t_curv,-std_t_curv]]
NLL_points_curv = NLL(np.array(points_curv),del_m_square)

points_dev = [min_theta_1D + x for x in [std_t_dev,-std_t_dev]]
NLL_points_dev = NLL(np.array(points_dev),del_m_square)

# NLL against theta    
plt.figure()
plt.title("NLL against $ \theta_{23}$ for $\Delta m^2 = 2.4e-3$")
plt.xlabel("Thetas")
plt.ylabel("NLL")
plt.plot(thetas, NLL_thetas,)
plt.plot(min_theta_1D,min_NLL_1D,'x',label = "Min NLL")
plt.plot(points_abs,NLL_points_abs,'x',label = "Absolute Error")
plt.plot(points_curv,NLL_points_curv,'x',label = "Curvature Error")
plt.plot(points_dev,NLL_points_dev,'x',label = "2nd Dev Curvature Error")
plt.legend()

#%%
# Two dimensional minimisation
def plot_mass(theta = np.pi/4):
    masses = np.linspace(0,5e-3,1000)
    NLL_masses = NLL(theta,masses)
    
    plt.figure()
    plt.plot(masses,NLL_masses)
    plt.xlabel("$\Delta m^2$")
    plt.ylabel("NLL")
    plt.title("NLL against delta mass squared")
#    plt.plot(0.0026026,NLL(np.pi / 4,0.0026026),'x')

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


    return min_x_mass,min_y_mass,min_x_thetas,min_y_t, mass_step,theta_step


# Obtain Univariate minimisation    
a,b,c,d,e,f = Univariate(func = NLL,guess_a = [1.5e-3,2e-3,3e-3],guess_b =[0.2, 0.5, 1])
min_masses_2D,min_NLL_masses,min_thetas_2D,min_NLL_thetas,mass_step,theta_step = (a,b,c,d,e,f)

# Find Minimum values of masses and thetas
min_mass_2D = min(min_masses_2D)
min_theta_2D = min(min_thetas_2D)

# Find errors on univariate
std_mass_2D = Error_Curvature(unoscillated,oscillated,min_masses_2D,
                      min_NLL_masses)
std_mass_t = Error_Curvature(unoscillated,oscillated,min_thetas_2D,
                      min_NLL_thetas)
    
print("Minimum mass 2D Univariate Minimiser = {:.7f} +/- {:.7f}".format(min_mass_2D,std_mass_2D))
print("Minimum theta 2D Univariate Minimiser = {:.4f} +/- {:.4f}".format(min_theta_2D,std_mass_t))
print("Minimum NLL 2D Univariate Minimiser = {:.4f}".format(min(min_NLL_thetas)))
#%%
plt.figure()
plt.plot(theta_step,mass_step)

plot_mass(theta = min(min_thetas_2D))
plt.plot(min(min_masses_2D),NLL(min(min_thetas_2D),min(min_masses_2D)),'x')

plot_theta(mass = min(min_masses_2D))
plt.plot(min(min_thetas_2D),NLL(min(min_thetas_2D),min(min_masses_2D)),'x')

#%% Plot Colour map

def plot_color_map(x,y,func):
    Z = []
    for i in range(len(x)):
        Z.append([func(x[j],y[i]) for j in range(len(y))])
    
    fig,axes = plt.subplots(1,1)
    
    c = axes.imshow(Z,origin="lower",interpolation = "None",aspect = "auto",
                    extent = [min(x),max(x),min(y),max(y)],cmap = 'plasma')
    
    plt.xlabel("$\Theta$")
    plt.ylabel("$\Delta m^2$")
    plt.title("NLL colour map of $\Delta m^2$ and $\Theta$")
    
    fig.colorbar(c,ax=axes, label='NLL')
#    print("\n",Z)
   
#plot_color_map(np.linspace(0,np.pi/2,100),np.linspace(0,5e-3,100),NLL)
plot_color_map(np.linspace(0,np.pi,100),np.linspace(0,5e-3,100),NLL)


#%% Simulated Annealing 
def Simulated_Annealing(func,N,T_start,T_step,xy_step,guesses,limit = None):
    
    def Thermal(E,T):
        k_b = 1.38e-23
        return np.exp(- E / (k_b * T))
    
    t_values = []
    
    x = [0]* N
    x_dash = [0] * N
    h = xy_step# step sizes (initial conditions)
    
    # Set guesses 
    for i in range(len(x)):
        x[i] = random.uniform(guesses[i][0],guesses[i][1])
    
    T = T_start
    success_count = 0
    count = 0
    
    theta_steps = []
    mass_steps = []
    
    while T > 0:

        if T % int(T_step * 1/(T_step/100)) == 0:
            print("Temperature {} K".format(T))
            
        for i in range(len(x)):
            x_dash[i] = random.uniform(x[i]-h[i],x[i] + h[i])
        
        if limit != None:
            
            while x_dash[0] >limit:
                x_dash[0] = random.uniform(x[0]-h[0],x[0] + h[0])

        
        del_f = np.array(func(*x_dash)) - np.array(func(*x)) 
        p_acc = Thermal(del_f,T)

        
        if p_acc > 1 : # if next energy value is smaller, then update
            theta_steps.append(x[0])
            mass_steps.append(x[1])

            for i in range(len(x)):
                x[i] = x_dash[i]
                if x_dash[i] < 1: #TODO: maybe 1 isn't the best value to use!
                    h[i] = x_dash[i]
                else: 
                    pass 
            
            success_count +=1
        else:
            pass 

        t_values.append(x_dash[0])
        T = round(T-T_step,10)      
        count+=1
        
        
#    print(np.unique(values))
    NLL_value = func(*x) 
    
    [print(x[i]) for i in range(len(x))] 
    print("Efficiency = {:.4f}%".format(success_count/count * 100))
    
    print(theta_steps)
    
    plt.plot(theta_steps,mass_steps)    
    plt.figure()
    NLL_theta = NLL(x[0],x[1])
    samp_x = np.linspace(0,2*np.pi,1000)
    plt.plot(samp_x,NLL(samp_x,x[1]))
    plt.plot(x[0], NLL_theta,'x')
    return #t_values


#%% 2D Simulated Annealing with NLL 
Simulated_Annealing(NLL,N = 2,T_start = 1000,T_step = 0.01,xy_step = [0.3,1e-3],
                    guesses = [[0.5,0.7],[1e-3,3e-3]],limit = np.pi/4)

#%% 3D Simulated Annealing with Cross Section
# 1000,0.01
Simulated_Annealing(NLL,N = 3,T_start = 1000,T_step = 0.01,
                    xy_step = [0.3,1e-3,0.2],guesses = [[0.7,0.8],[1e-3,3e-3],[0.5,1]]
                    ,limit = np.pi/4)
#    
#%% Test Functions

def parabola(x,y):
    x = np.array(x)
    y = np.array(y)
    return (x-1)**2 + (y-1)**2

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
print(test_x)

test_x, test_y = Parabolic(guess_x = [-5,0,5],func = parabola,param = "1D_general")
print(test_x)

# Univariate minimisation
a,b,c,d = Univariate(func = parabola,guess_a = [-0.5,0,0.5],guess_b = [-0.5,0,0.5])
print("h343",min(a),min(c))

a,b,c,d = Univariate(func = Ackley,guess_a = [-0.5,0,0.5],guess_b = [-0.5,0,0.5])

print("h343",min(a),min(c))

# Simulated Annealing test with Ackley function 
Simulated_Annealing(Ackley,N = 2,T_start = 1000,T_step = 0.01,xy_step = [-5,5],
                      guesses = [[-5,5],[-5,5]])


#%%
    #TODO: comment why do m first -- plot graphs
"""

"""
    #TODO: Comment on merit of both methods of error
"""

"""
    #TODO: Comment on annealing better than univariate?
"""

"""
    #TODO: Comment on cross section?
    #TODO: comment Error as pi/4?
    
    #TODO: Verify step sizes of temperature, 
    #TODO: Uncertainty on mass? on theta? from simulated annealing
