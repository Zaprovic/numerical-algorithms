import numpy as np
from roots import bisection

f = lambda x: np.arctan(np.exp(x)) - x / (x-2) 

a = -1.5
b = 0

r = bisection.bisection(f,a,b,1e-5)
print(r)