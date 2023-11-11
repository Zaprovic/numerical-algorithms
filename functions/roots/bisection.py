import numpy as np
import pandas as pd


def helper():
    return '''Bisection method for transcendental equations
    
Function : bisection(f,s a, b , tol)
f : Function must be created as a Python function or using lambdas
a : Left side of the interval
b : Right side of the interval
tol : Maximum desired tolerance for the output
'''


def bisection(f, a, b, delta):
    if f(a)*f(b) > 0:
        raise ValueError("There is no root in the interval")

    i = 0
    fa = f(a)
    n0 = int(np.floor(np.log2((b-a)/delta)))
    data = []

    while i <= n0:
        p = a + (b-a)/2
        fp = f(p)

        data.append([a, b, p, fp, i+1])

        if fp == 0 or (b-a)/2 < delta:
            return pd.DataFrame(data, columns=['a', 'b', 'p', 'f(p)', 'Iteration'])

        i += 1

        if fa*fp > 0:
            a = p
            fa = fp
        else:
            b = p

    return f'Method failed after {n0} iterations'


