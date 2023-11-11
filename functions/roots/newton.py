import numpy as np
from scipy.misc import derivative


def helper():
    return '''Newton's method for transcendental equations
    
Function : newton(f, x0, delta, epsilon, m)
f : Function must be created as a Python function or using lambdas (make sure to use sympy instead of numpy to declare special functions such as exp, sin, etc)
x0 : Initial approximation of the root
delta : Maximum absolute error for the root (|pn+1 - pn| < delta)
epsilon: Maximum error that satisfies f(pn) < epsilon
m : Maximum number of iterations
'''


def newton(f, x0, delta, epsilon, m):
    df = lambda x: derivative(f, x, dx=1e-6)

    for i in range(m + 1):
        p = x0 - f(x0) / df(x0)
        e = np.abs(p - x0)
        x0 = p
        y = f(p)

        if (np.abs(y) < epsilon) or (e < delta):
            return f'Root: {p}\nAbsolute error: {e} \nIteration: {i + 1}\nf(root): {y}'

    return 'Max number of iterations exceeded'
