#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 19:10:25 2019

@author: ShaunGan
"""
## Change Working Directory if data.txt not found!

# import os
# os.chdir("/Users/ShaunGan/Desktop/computational-physics/Neutrinos")
 #%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange,interp1d
from numpy.polynomial.polynomial import Polynomial
import random
import warnings

from plot_functions import plot_mass,plot_theta,plot_color_map,plot_steps
from plot_functions import plot_rates,plot_std, plot_unosc

#%% 3.1 The data
def read_data(filename):
    """
    pandas is used to read the raw text file and output it as a pandas 
        dataframe containing 3 columns: unoscillated rate, oscillated (measured)
        rate, and energies,
    -----------
    Parameters:
    -----------
    filename: is the file name 
    """
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
# Read Data
data = read_data("data.txt")

# Set Guess Parameters
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

#%% 3.2 Fit Function
def survival_probability(E,theta,del_mass_square,L):
    """ 
    Function for the survival probability of neutrino oscillations
    -----------
    Parameters:
    -----------
    E: energies bins
    theta: mixing angle
    del_mass_square: delta mass square
    L: distance travelled, fixed at 295km. 
    """
    coeff = 1.267 * del_mass_square * L  / E
    P = 1 - np.sin(2*theta)**2 * np.sin(coeff) **2
    return P

#%%
def cross_section(E,a,C):
    """
    Function for the cross section
    -----------
    Parameters:
    -----------
    E: Energies, 
    a: alpha ~ cross section rate [d(cross section)/dE]
    C: offset, set to zero. 
    """
    return C + a * E    

def oscillated_prediction(thetas,masses, cross_a = "None",C = 0):
    """
    Predicts oscillation rates from unoscillated rates, by multiplying 
        survival probability with unoscillated rates. 
        
    Conditional statements implemented to return a singular value or list of 
        elements, depending on the nature of the input. Applied for all 3 
        parameters. 
    -----------
    Parameters:
    -----------
    thetas: mixing angles
    masses: delta mass square values
    cross_a: cross section rate
    C: offset, set to zero
    """
    
    # Calculate probabiliites of oscillation
    
    # varying mass with other parameters constant
    if np.array(masses).size > 1: 
        probs = [survival_probability(energies, thetas, masses[i], L) for i in
                 range(len(masses))]        
        
    # varying theta with other parameters constant    
    elif np.array(thetas).size > 1:
        probs = [survival_probability(energies,thetas[i],masses,L) for i in
                 range(len(thetas))]
    
    # both theta, mass constant      
    else:  
        probs = [survival_probability(energies,thetas,masses,L)]     
  
    # Obtain unoscillated rates from data grame
    unosc_rates = data["unoscillated_rate"].tolist()
    
    # Convert to numpy arrays for element wise operations
    probs = np.array(probs)
    unosc_rates = np.array(unosc_rates)
    
    # Calculate cross sections
    if cross_a == "None":
        cross_sections = 1 #Returns to the original parameters
    
    elif np.array(cross_a).size > 1:
        cross_sections = [cross_section(energies,cross_a[i],C) for i in 
                          range(len(cross_a))]
    else:
        cross_sections = cross_section(energies,cross_a,C)

    # Find oscillated rates
    osc_rates = probs * unosc_rates * cross_sections
    return osc_rates

#%% 3.2 Data Exploration -- Histograms
del_m_square = 2.6e-3 # adjusted to fit code data better

predicted = oscillated_prediction(t_IC,del_m_square)
plot_unosc(energies,unoscillated,oscillated)
plot_rates(energies,oscillated,predicted,[t_IC,del_m_square,0,0])

#%% 3.3 Likelihood Function
def NLL(theta_values,del_m,cross_a = "None",cross_C = 0):
    """
    Negative log-likelihood function. Taking the minimum of this function will
        provide model values for the parameters describing the neutrino
        oscillation survival probabilities. 
    -----------
    Parameters:
    -----------
    theta_values: mixing angles  
    del_m: delta mass square
    cross_a: cross section rate
    
    --> Values above can be integers, floats, lists or np.arrays.
    
    C: offset, set to zero
    """
    rates = oscillated_prediction(thetas=theta_values,masses=del_m,
                                  cross_a=cross_a,C = cross_C)
    k = oscillated
    
    NLL_value = []
    for j in range(len(rates)):
        tmp = [] # Temporary list to find sum 
        l  = rates[j]
        for i in range(len(l)):
            
            #if measured rates not zero, calculate log            
            if k[i] != 0: 
                value = l[i] - k[i] + k[i] * np.log(k[i]/l[i])
                tmp.append(value)
            
            # if measured rates zero, klog(k) --> 0, so ignore. 
            else:
                value = l[i]
                tmp.append(value)

        NLL_value.append(sum(tmp)) # Calculate sum of all values 
    
    #Return value depending on float/ list nature of inputs. 
    if len(NLL_value) ==1:
        return NLL_value[0]
    else:
        return NLL_value

#%% Estimate Minimum of Parameters
plot_mass(func = NLL,theta = t_IC)
plot_theta(func = NLL)

#%% 3.4 Minimise
 
def Parabolic(guess_x,IC = None,func = None,param = 'theta',
              return_smallest = False):
    """
    Parabolic minimiser to obtain the minimum of a function. Starting from 
        three points, generates a new x_3 of lower f(x), keeps the three 
        x vales with the lowest f(x) values. Iterates until convergence, 
        where values found are constant at 15 decimal places. 
    
    If mixture of positive ang negative curvature, two additional points
        generated between the range. The three points out of the five
        with the lowest f(x) values are kept. 
    
    Based on the intergral of the lagrange polynomial. 

    -----------
    Parameters:
    -----------  
    guess_x: list of three values, as initial guesses 
    IC: Initial conditions, L and energies
    func: function to minimise, NLL or test functions
    param: 
        'theta' - minimise in theta for NLL
        'mass' - minimise in mass for NLL
        '1D_general" - minimse in an arbitary parameter for arbitary function
    return_smallest: returns the latest generated x_3 value if True,
                        if False, returns last three points.
    """

    def _find_next_point(x_list, y_list):
        """
        Finds x_3, from x_{1,2,3} and y_{1,2,3}. Based on Lagrange Polynomials
        -----------
        Parameters:
        -----------  
        x_list,y_list: list of x and y, each of 3 elements
        """
        # Extract x and y values
        x_0, x_1, x_2 = tuple(x_list)
        y_0, y_1, y_2 = tuple(y_list)
        
        # Calculate numerator and denominator
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
    
    # Allows you to minimise in differen parameters, within one Parabolic func
    if param == "theta":
        Func = lambda k: func(np.array(k),IC)
    elif param == "mass":
        Func = lambda k: func(IC,np.array(k))
    elif param == "1D_general":
        y = 0
        Func = lambda k: func(k,y)
    else:        
        raise ValueError("Invalid param type")
    
    # Rename variables, and calculate NLL variables for 3 x-values
    vals_x = guess_x
    vals_y = Func(vals_x)

    # Find next x_3 point 
    x_3 = _find_next_point(vals_x,vals_y)
    x_3_last = x_3 + 10
    
    # Append x_3 point and find f(x) for 4 x-values
    vals_x.append(x_3)
    vals_y = Func(vals_x)
    
    # Used to assess convergence 
    end_count = 0
    
    while end_count<5:  
        # Sort values. 
        vals_x.sort()  
        
        # Find maximum f(x) values
        max_idx = np.argmax(vals_y)
        
        # delete the smallest x value
        del vals_x[max_idx]
                
        # Finds the new f(x) values
        vals_y = Func(vals_x)

        # Finds the next minimum value
        x_3_last = x_3
        x_3 = _find_next_point(vals_x, vals_y)

        # Check for mixture of negative and positive curvature
        if Func(x_3) > all(vals_y):
            
            # Warn about positive and negative curvatures
            warnings.warn("Interval has positive & negative curvature", Warning) 

            vals_x.append(x_3)

            # finds 2 additional values from max and min of interval
            x_values = np.linspace(min(vals_x),max(vals_x),4)[1:3]
            x_values = np.append(x_values,vals_x)
            
            # finds f(x)
            y_values = list(Func(x_values))
            
            # Gets indices of a sorted array
            indices = np.argsort(y_values)
            
            #picks the 4 smallest values to carry on with
            vals_x = [x_values[i] for i in indices][0:4]
            vals_y = [y_values[i] for i in indices][0:4]

        else:
            # Stores the next smallest value
            vals_x.append(x_3)
            
            # Calculates the f(x) of four x values
            vals_y = Func(vals_x)
        
        # If values are constant to 15 decimal places five times, means converged
        if round(x_3,15) == round(x_3_last,15):
            end_count +=1
        else:
            end_count = 0
            
    if return_smallest == True:
        return x_3
    else:
        # Deletes largest x-value, and returns array of three values. 
        max_idx = np.argmax(vals_y)
        del vals_x[max_idx]      
        vals_y = Func(vals_x)
    
        return vals_x,vals_y
#%%
# Parabolic Minimise
vals_x,vals_y = Parabolic(IC=del_m_square,guess_x = [0.2,0.4,0.6],func = NLL)

# Minimums based on the parabolic minimiser
min_theta_1D = min(vals_x)
min_NLL_1D = min(vals_y)    

print("1D Minimisation, theta = {:.4f}, NLL = {:.4f}".format(min_theta_1D,min_NLL_1D))

#%% 3.5 Find Accuracy of Result

def Error_abs_addition(min_p,min_NLL,func = NLL):
    """
    Finds error on minimum from NLL fit, using principle that uncertainty
        is determinined by + 0.5 from the minimum
    
    Points are linearly interpolated in order to extract value of x at 
        minimum NLL + 0.5
        
    Function extended to provide error in 2D and 3D. In higher dimensions,
        only one variable is changed with the error found, whilst the other
        values are kept constant. 
        
    -----------
    Parameters:
    -----------      
    min_p: list of minimum values from each parameter, accepts list up to 3D
    min_NLL: float value of the minimum calculated NLL value
    func: function, set to NLL
    """
    
    # Number of points for interpolation
    acc = 1000 
    
    # Ranges to search, so that it includes NLL + 0.5
    theta_range = 0.7
    mass_range = 1e-3
    cross_range = 0.5
    above_min = 0.7
    
    # If theta is above pi/4 shift it back by pi/2 and take abs value
    if min_p[0] > np.pi/4:
        min_p[0] = abs(min_p[0] - np.pi/2)

    # Generate array values for thetas and masses, to interpolate
    thetas = np.linspace(min_p[0]-theta_range,np.pi/4,acc)
    masses = np.linspace(min_p[1]-mass_range,min_p[1]+mass_range,acc)
    
    # 2D
    if len(min_p) == 2:
        
        # Generate functions 
        NLL_func_t = lambda k: func(np.array(k),min_p[1])
        NLL_func_m = lambda k: func(min_p[0],np.array(k))
        
        # Create list of functions and values
        funcs = [NLL_func_t,NLL_func_m]
        values = [thetas,masses]

    # 3D
    elif len(min_p) == 3:
        
        # Generate array values for cross sections,  to interpolate
        crosses = np.linspace(min_p[2]-cross_range,min_p[2]+cross_range,acc)
        
        # Generate functions 
        NLL_func_t = lambda k: func(np.array(k),min_p[1],min_p[2])
        NLL_func_m = lambda k: func(min_p[0],np.array(k),min_p[2])
        NLL_func_c = lambda k: func(min_p[0],min_p[1],np.array(k))
        
        # Create list of functions and values
        funcs = [NLL_func_t,NLL_func_m,NLL_func_c]
        values = [thetas,masses,crosses]
        
    else:
        raise ValueError("Incorrect Dimensions")
        
    # Empty list to store stds. 
    stds= []
    
    for i in range(len(min_p)) :
        # Make Dataframe for each variable, for easier data manipulation
        df = pd.DataFrame({"Var":values[i],"NLL":funcs[i](values[i])})
        df = df.loc[(df.NLL < min_NLL + 2*above_min)]

        # Locate values on the LHS and RHS of the minimum
        LHS = df.loc[df.Var < min_p[i]]
        RHS = df.loc[df.Var > min_p[i]]
                
        # Interpolate to find continous & exact value
        func_LHS = interp1d(LHS["NLL"],LHS["Var"]) # Flipped to inverse function
        func_RHS = interp1d(RHS["NLL"],RHS["Var"]) # Flipped to inverse function
        
        # Finds list of upper and lower error
        std_dev = np.array([func_LHS(min_NLL + 0.5)-min_p[i],func_RHS(min_NLL + 0.5)-min_p[i]])
        
        # Stores in list stds, containing errors for all parameters. 
        stds.append(list(std_dev))
        
    return stds

#%%
def Error_Curvature(unoscillated_rates,measured_events,parabolic_x,parabolic_y): 
    """
    Fits a 2nd order Lagrange polynomial Ax^2 + Bx +C, where the curvature A
        is assessed to find the values of min NLL + 0.5.
        
    The equation Ax^2 = 0.5 is solved to find the values of x, as Bx and C
        are both translational variables.
        
    -----------
    Parameters:
    -----------   
    unoscillated_rates: unoscillated rates from the raw data file
    measured_events: oscillated measured data
    parabolic_x: final three x values from the parabolic minimiser
    parabolc_y: f(parabolic_x), where f would be the NLL function
    """
    # Calculate second derivative of Lagrange polynomial fit
    x = parabolic_x
    y = parabolic_y
    
    # Based on the parabolic minimiser
    poly = lagrange(x,y)
    coefficients = Polynomial(poly).coef
    
    # Just consider curvature, ax**2 = 0.5
    std_theta = np.sqrt(0.5/coefficients[0])
    
    return std_theta

#%%
def Error_2nd_Dev(unoscillated_rates,measured_events,del_mass_square,IC,parabolic_x,parabolic_y):
    """
    Calculates error by finding difference in second derivatives. For this
        example, this was implemented with respect to theta, however as just
        an indicative measure, the parameter chosen should not matter. 
        
    To be used generally, the second derivative for each parameter would need to
        be found. 
        
    -----------
    Parameters:
    ----------- 
    unoscillated_rates: unoscillated rates from the raw data file
    measured_events: oscillated measured data
    parabolic_x: final three x values from the parabolic minimiser
    parabolc_y: f(parabolic_x), where f would be the NLL function
    del_mass_square: delta mass square
    IC: L, energies
    """
    # Find minimum theta and minimum NLL
    min_val = min(parabolic_x)
    
    # Extract values
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
    
    return curvature_del

#%%

# Calculate error form + 0.5 through linear interpolation. 
std_t_abs = Error_abs_addition([min_theta_1D,del_m_square],min_NLL_1D)
std_t_abs = std_t_abs[0]

# Calculate error form + 0.5 through parabolic interpolation. 
std_t_curv = Error_Curvature(unoscillated,oscillated,vals_x,vals_y)

# Calculate error from + 0.5 through fitting to second derivative  
std_t_dev = Error_2nd_Dev(unoscillated,oscillated,del_m_square,ICs,vals_x,vals_y)

# Plot Errors
NLL_thetas = NLL(thetas,del_m_square)
plot_std(min_theta_1D,del_m_square,std_t_abs,std_t_curv,NLL,thetas,NLL_thetas,min_NLL_1D)

print("===========================")
print("From 1D Parabolic Minimiser")
print("===========================")
print("Absolute Error: Min Theta = {:.6f} +/- {}".format(min_theta_1D,std_t_abs))
print("Curvature Error: Min Theta = {:.6f} +/- {:.6f}".format(min_theta_1D,std_t_curv))
print("2nd Dev Error: Change in Curvature = {}".format(std_t_dev))

#%% Comment Error as theta --> pi/4
"""
    As the value of theta tends to pi/4, the bump seen at pi/4 reduces and 
    theta tends to a minimum value, and hence errors at this point would be 
    symmetrical. 
    
    The bump does always reduce as theta --> pi/4, avoiding the incorrect case 
    where points are interpolated in a manner that neglects the local minima.
"""

#%% Comment on merit of both methods of error
"""
    The curvature method provided a more precise uncertainty for the parabolic 
    minimiser as it dealt specifically with parabolas. 
    
    The linear fit requires less paramters for implementation, but depends on 
    the accuracy used between points when interpolated. Although with a 
    sufficiently high accuracy, the difference between methods would be small 
    (at the cost of computation time).
    
    Both came out to give similar results within 1\% of each other. 
    
    Curvature was used for Univariate and linear was used for Simulated 
    Annealing. 
"""

#%% 4.1 Univariate Method

def Univariate(func,guess_a,guess_b):
    """
    Univariate Parabolic Minimiser method, applies the parabolic method in
        1D to each parameter in turn, until convergence is achieved.
    Iterates until convergence, where values found are constant at 15 decimal 
        places.     
    -----------
    Parameters:
    ----------- 
    func: function
    guess_a, guess_b: list of three values, as initial guesses
    """
    # Rename variables
    min_x_mass = guess_a
    min_x_thetas = guess_b
    
    # Find Radius
    R = np.sqrt(min(min_x_thetas)**2 + min(min_x_mass)**2)
    R_last = R + 10 # Arbitrary starting values
    
    # Used for convergence
    end_count = 0
    
    # Store minimum value to show steps taken by univariate
    mass_step = [min(min_x_mass)]
    theta_step = [min(min_x_thetas)]
    
    while end_count <5:        
        
        # Minimise mass first
        min_x_mass,min_y_mass = Parabolic(func = func,IC=min(min_x_thetas), guess_x=min_x_mass,param = "mass")
        
        # Store steps to show steps taken by Univariate
        mass_step.append(min(min_x_mass))
        theta_step.append(min(min_x_thetas))

        # Minimise theta next        
        min_x_thetas, min_y_t = Parabolic(func = func,IC=min(min_x_mass), guess_x=min_x_thetas)
        
        # Store steps to show steps taken by Univariate
        mass_step.append(min(min_x_mass))
        theta_step.append(min(min_x_thetas))
            
        # Finds if value repeats 3 times
        R = np.sqrt(min(min_x_thetas)**2 + min(min_x_mass)**2)                
        if round(R,18) == round(R_last,18):
            end_count +=1
        else:
            end_count = 0
        R_last = R
        
    return [min_x_mass,min_y_mass,min_x_thetas,min_y_t, mass_step,theta_step]

#%% Obtain Results

# Obtain Univariate minimisation    
values = Univariate(func = NLL,guess_a = [1.5e-3,2e-3,3e-3],guess_b =[0.2, 0.5, 1])

# Relabel values 
min_masses_2D,min_NLL_masses,min_thetas_2D,min_NLL_thetas,mass_step,theta_step = tuple(values)

# Find Minimum values of masses and thetas
min_mass_2D = min(min_masses_2D)
min_theta_2D = min(min_thetas_2D)

# Find errors on univariate, using curvature
std_mass_2D = Error_Curvature(unoscillated,oscillated,min_masses_2D,
                      min_NLL_masses)
std_t_2D = Error_Curvature(unoscillated,oscillated,min_thetas_2D,
                      min_NLL_thetas)

# Find errors on 2nd Derivative, i.e. curvature
std_t_dev_2D = Error_2nd_Dev(unoscillated,oscillated,min_mass_2D,ICs,
                          min_thetas_2D,min_NLL_thetas)

print("============================")
print("From 2D Univariate Minimiser")
print("============================") 
print("Minimum theta = {:.4f} +/- {:.4f}".format(min_theta_2D,std_t_2D))
print("Minimum mass = {:.7f} +/- {:.7f}".format(min_mass_2D,std_mass_2D))
print("Minimum NLL = {:.4f}".format(min(min_NLL_thetas)))
print("2nd Dev Error: Change in Curvature = {}".format(std_t_dev_2D))

#%% Comment why do m first -- plot graphs
"""
    Delta m square was minimised before minimising theta. 
    
    This is because NLL against delta m square displayed many more local minima 
        when theta was near pi/4. Setting a low guess theta first (~0.1), 
        produces a smoother function for NLL against delta m square, reducing 
        the chances of getting stuck in a local minima, increasing the speed of 
        the method. 
        
"""
#%% Verify Univariate minimum values using plots

# Plot steps taken by Univariate
plot_color_map(np.linspace(0,np.pi,100),np.linspace(0,5e-3,100),NLL)
plot_steps(theta_step,mass_step,"$\Theta$","$\Delta m^2$",
           "Accepted steps for Univariate minimisation of NLL",font_size = "small")
plt.legend()

# Plots minimum value on NLL against del m square
plot_mass(func = NLL,theta = min(min_thetas_2D),val = round(min(min_thetas_2D),4))
plt.plot(min(min_masses_2D),NLL(min(min_thetas_2D),min(min_masses_2D)),'x')

# Plots minimum value on NLL against theta
plot_theta(func= NLL,mass = min(min_masses_2D),val = round(min(min_masses_2D),6))
plt.plot(min(min_thetas_2D),NLL(min(min_thetas_2D),min(min_masses_2D)),'x')

# Plot Rates against Energy, to see in predicted oscillated value matches measured data
predicted = oscillated_prediction(min_theta_2D,min_mass_2D)
plot_rates(energies,oscillated,predicted,[min_theta_2D,min_mass_2D ])

#%% 4.2 Simultaneous Minimisation 

def Simulated_Annealing(func,N,T_start,T_div,xy_step,guesses,PCS_mode = True,
                        PE =0.3, PL = 0.29):
    """
    Simulated Annealing is a probabilistic minimisation method based on the 
        principle of thermodynamics. It is governed by the Boltzmann 
        probability distribution, where random fluctuations could push the 
        system into a states of higher energy, corresponding to a local minima. 
        This allows the procedure to escape from local minima and search for 
        the global minimum.
    
    This function is an N dimensional Simulated Annealing function, where each
        parameter is stored in a list. 
        
    -----------
    Parameters:
    ----------- 
    func: Function
    T_start: Starting Temperature
    T_div: Sets number of temperature steps taken to reduce T to zero
    xy_step: list of step sizes for each parameter
    guesses: list of lists of two values indicating the range of the value
    PCS_mode: if True, using cooling scheme. If False, uses uniform steps
    PE, PL: constants controlling cooling rates of PCS
    """
    
    def PCS(T_o,T_f,T_div,i):
        """
        A Probabilistic Cooling Scheme was used to cool the temperature. 
        
        The function contains a log and exponential component that combine 
        to cool the temperature at different rates depending on its value. 
        
        At high temperatures, the exponential component dominates the equation, 
        causing $T$ to reduce very quickly, whereas the logarithmic component 
        dominates low temperatures allowing for smaller increments. 
        
        This causes a quick search across all local minimas, and an accurate 
        search for the global minima.         
        
        -----------
        Parameters:
        ----------- 
        T_o: Initial Temperature
        T_f: Final Temperature
        i: Step number
        """
        N = abs(T_f - T_o) / T_div
        a = 0.01
        
        A = (T_o - T_f)*(N+1)/N
        B = T_o - A
        T_i = (A/(i+1)+B)*PE + (a*T_o/np.log(1+i))*PL
        return T_i

    # Boltzmann Equation
    def Thermal(E,T):
        k_b = 1# Setting to 1 allows for reasonable values for p_acc.
        return np.exp(- E / (k_b * T))

    # Create list to track steps taken my Sim-Anneal
    t_steps = []
    m_steps = []
    
    # Create list of zeros, to allow for N dimensional minimisation
    x = [0]* N
    x_dash = [0] * N
    
    # step sizes (initial conditions)
    h = xy_step
    
    # Set guesses 
    for i in range(len(x)):
        x[i] = random.uniform(guesses[i][0],guesses[i][1])
    
    # Store counters to find efficiency of method
    success_count = 0
    count = 1
    
    # Relable starting temperature
    T = T_start
    print("Starting Temperature {:.2f} K".format(T))
    
    # Just larger than machine epsilon ~ 1e-16
    while T > 1e-15:

        # Print temperature to track progress
        if count % 1000 == 0:
            print("Temperature {:.4f} K".format(T))
            
        # Generate new points for each parameter
        for i in range(len(x)):
            x_dash[i] = random.gauss(x[i],h[i])
        
        # Find change in energy from new point and old point 
        del_f = np.array(func(*x_dash)) - np.array(func(*x)) 
        
        # Find acceptance probability
        p_acc = Thermal(del_f,T)
       
        # if next energy value is smaller, then kepp 100% of time
        if del_f <= 0:
            for i in range(len(x)):
                x[i] = x_dash[i]    
            
            # Keep track of steps
            success_count +=1
            t_steps.append(x[0])
            m_steps.append(x[1])
        
        # accept value with probability, to escape local minimas 
        elif p_acc > random.uniform(0,1):
            for i in range(len(x)):
                x[i] = x_dash[i]  
                
            success_count +=1
            t_steps.append(x[0])
            m_steps.append(x[1])
            
        else:
            pass 
        
        # Option to turn off PCS
        if PCS_mode == False:
            T -= T_div
        else:
            T = PCS(T_start,0,T_div,count)
        count+=1
        
    # Append final value
    t_steps.append(x[0])
    m_steps.append(x[1])

    # Find NLL Value
    NLL_value = func(*x) 
    print("Minimum function value = ",NLL_value)
    print("Minimum Values = ",x)

    print("Efficiency = {:.4f}% of {} steps".format(success_count/count * 100,
                                                      count))    
    return x, NLL_value, [t_steps,m_steps]



#%% Sample Run - 2D Simulated Annealing with NLL 
min_p,min_NLL,steps = Simulated_Annealing(NLL,N = 2,T_start = 1000,T_div = 1,
                                      xy_step = [0.1,0.5e-3],
                                          guesses = [[0.7,0.8],[2e-3,3e-3]])

# Find errors
stds = Error_abs_addition(min_p,min_NLL)

print("===========================")
print("From 2D Simulated Annealing")
print("===========================") 
print("Minimum theta = {:.4f} + {}".format(min_p[0],np.round(stds[0],4)))
print("Minimum mass = {:.7f} + {}".format(min_p[1],np.round(stds[1],7)))
print("Minimum NLL = {:.4f}".format(min_NLL))

#%% Plot rates against energy to see if parameters fit measured data
predicted = oscillated_prediction(*min_p)
plot_rates(energies,oscillated,predicted,min_p)

#%% 5. Neutrino Oscillation cross-section

# Sample Run - 3D Simulated Annealing with Cross Section
min_p,min_NLL,steps = Simulated_Annealing(NLL,N = 3,T_start = 1000,T_div = 1,
                                          xy_step = [0.15,0.5e-3,0.1],
                                          guesses = [[0.4,1],[1e-3,5e-3],
                                                     [0.5,2.5]])
# Find errors          
stds = Error_abs_addition(min_p,min_NLL)

print("===========================")
print("From 3D Simulated Annealing")
print("===========================") 
print("Minimum theta = {:.4f} + {}".format(min_p[0],np.round(stds[0],4)))
print("Minimum mass = {:.7f} + {}".format(min_p[1],np.round(stds[1],7)))
print("Minimum cross section = {:.4f} + {}".format(min_p[2],np.round(stds[2],4)))
print("Minimum NLL = {:.4f}".format(min_NLL))

#%% Plots to verify 3D Simulated Annealing

#Plot rates against energy to see if parameters fit measured data
predicted = oscillated_prediction(*min_p)
plot_rates(energies,oscillated,predicted,min_p)

# Plot steps for 3D Simulated Annealing
plot_color_map(np.linspace(0,np.pi/2,100),np.linspace(0,5e-3,100),NLL)
plot_steps(steps[0],steps[1],"$ \Theta_{23}$","$\Delta m_{23}^2$",
           "Accepted Steps for Simulated Annealing for NLL")
plt.legend(loc=1, prop={'size': 12})

#%% Comment on cross section
"""
    The NLL value was seen to reduce by 81% when comparing between the 2D and 
    3D case. 
    
    This enforces the idea that the cross section is a suitable parameter to 
    neutrino oscillations.
"""
#%% Comment on annealing better than univariate
"""
    For comparison, 
    
    Simulated Annealing requires a longer time but can have a 
    large search range before converging on the global minima of the system. 
    
    Univariate is much quicker and direct in its search since it is not 
    probabilistic, allowing results to be reproducible. However it can get 
    trapped in local minima and thus benefits from a small starting search 
    range.
"""

#%% 4D minimisation, with offset, to verify. 
min_p,min_NLL,steps = Simulated_Annealing(NLL,N = 4,T_start = 1000,T_div = 1,
                                          xy_step = [0.15,1e-3,0.2,0.05],
                                          guesses = [[0.7,0.8],[2e-3,3e-3],
                                                     [0.7,0.8],[0.4,0.5]])
# Returns C ~ 0
            
#%% Test Function: Ackley
            
def Ackley(x,y):
    """
    Ackely Function, created to test minimisers. 
    """
    x = np.array(x)
    y = np.array(y)
    A = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))
    B = - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
    return A + B + np.exp(1) + 20

# Generate sample data to display plot
sample_x = np.linspace(-5,5,1000)
plt.figure()
plt.plot(sample_x, Ackley(sample_x,0))
plt.title("Ackley Function")

# Parabolic minimisation
test_x, test_y = Parabolic(guess_x = [-5,0,5],func = Ackley,param = "1D_general")

# Univariate minimisation
test = Univariate(func = Ackley,guess_a = [-0.5,0,0.5],guess_b = [-0.5,0,0.5])

# Simulated Annealing Minimisation
p,y,steps = Simulated_Annealing(Ackley,N = 2,T_start = 1000,T_div =1,
                                          xy_step = [0.2,0.2],
                                          guesses = [[-5,5],[-5,5]],
                                          PE = 0.3,PL = 0.29)

# Display Results
print("\n1D Parabolic Min for Ackley x = {} where (y = 0)".format(min(test_x)))
print("2D Univariate Min for Ackley: x = {},y = {}".format(min(test[0]),
      min(test[2])))
print("2D Simulated Annealing: x = {:.4f},y = {:.4f}".format(p[0],p[1]))

# Plot colour map
plot_color_map(np.linspace(-10,10,200),np.linspace(-10,10,200),Ackley)
plot_steps(steps[0],steps[1],"x","y","Simulated Annealing for Ackley")
plt.legend(loc=1, prop={'size': 12})

#%% Test Function: Sphere
def sphere(x,y):
    x = np.array(x)
    y = np.array(y)
    return (x-1)**2 + (y)**2

# Parabolic minimisation
test_x = Parabolic(guess_x = [-5,0,5],func = sphere,param = "1D_general",
                           return_smallest = True)
# Univariate minimisation
test = Univariate(func = sphere,guess_a = [-0.5,0,0.5],guess_b = [-0.5,0,0.5])

# Display Results
print("2D Univariate Min for sphere: x = {},y = {}".format(min(test[2]),min(test[1])))
print("1D Parabolic Min for 2D sphere x = {} where (y = 0)\n".format(test_x))



