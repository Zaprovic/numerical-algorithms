import matplotlib.pyplot as plt
import numpy as np


def heun(f,a,b,alpha,n):
    h = (b-a) / n
    
    t = np.linspace(a,b,n+1)
    w = np.zeros(n+1)
    w[0] = alpha
    
    for i in range(n):
        w[i+1] = w[i] + (h/4) * (f(t[i],w[i]) + 3*f(t[i] + 2*h/3, w[i] + (2*h/3) * f(t[i] + h/3,w[i] + (h/3) * f(t[i], w[i]))))
    
    plt.grid()     
    if h > 0.1:
        plt.scatter(t,w, c='r')
        plt.plot(t,w)
        plt.show()
        
    else:
        plt.plot(t,w)
        plt.show()

    return w

f = lambda t,y: y - t**2 + 1
a = 0 
b = 2
alpha = 0.5
n = 10

S = heun(f,a,b,alpha,n)  
print(S)