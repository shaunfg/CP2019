import numpy as np
import matplotlib.pyplot as plt
#%%

n = 100

def Lagrange(n_values,func):
    """
    Find the product for each j first, and then sum all the ones together
    
    Comments:
        Seems to tail off at large values of x at large n, around n > 21
    """
    x = np.linspace(0,100,n_values)
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
#    plt.figure()
    plt.plot(x,y,label = "Original")
    plt.xlim(0,60)
    plt.ylim(1,-1)
    plt.legend()
    print(plot_func)
    
Lagrange(n,np.sin)

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