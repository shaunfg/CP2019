import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange

#%%

n = 23

def Lagrange(n_values,func):
    """
    Find the product for each j first, and then sum all the ones together
    
    Comments:
        Seems to tail off at large values of x at large n, around n > 21
    """
    max_value = 10
    
    x = np.linspace(0,max_value,n_values)
    y = func(x)
    
    final_poly = [0]
    
    for i in range(n_values):
        poly = [1] 
        for j in range(n_values):
            value_1 = i
            value_2 = j
            if value_1 == value_2:
                pass
            else: 
                denominator = x[i] - x[j]
                coeff = [x/denominator for x in [1,-x[j]]]
                poly = np.polymul(poly,coeff)
                
        poly = poly * y[i]
        final_poly = np.polyadd(final_poly,poly)
       
#        print("\n")
#        print(final_poly)
        
    plot_func = np.poly1d(final_poly)
    plt.plot(x,plot_func(x),label = "Lagrange")
    print(plot_func)

    x_values = np.linspace(0,max_value,100)
    y_values = func(x_values)
    plt.plot(x_values,y_values,label = "Original")
    poly = lagrange(x, y)
    plt.plot(x,poly(x),label = "scipy")
    plt.legend()
    plt.title("Sin plot - Lagrange Polynomials")
    
    
Lagrange(n,np.sin)

#%%

from scipy.interpolate import lagrange
x = np.array([0, 1, 2])
y = x**3
poly = lagrange(x, y)
print(poly)

from numpy.polynomial.polynomial import Polynomial
Polynomial(poly).coef

#%%

print(np.poly1d([4,3,4]))
foo =np.polyadd([2,3],[5,6,7],[3,3])
print(np.poly1d(foo))

#%% For loop POC

for i in range(5):
    for j in range(5):
        value_1 = i
        value_2 = j
        if value_1 == value_2:
            pass
        else: 
            print("x - {}/ {}- {}".format(value_2,value_1,value_2))
    print("\n")