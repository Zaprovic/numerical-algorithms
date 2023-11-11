import numpy as np


def helper():
    return '''Fixed point iteration method for transcendental equations in the form of x = g(x)
    
Function : fixedpt(g, p0, tol, m)
g : Function must be created as a Python function or using lambdas
p0 : Initial guess for the root
tol : Maximum allowed tolerance
m : Maximum numbers of iterations
'''


def fixedpt(g, p0, tol, m):

    P = [p0]

    for k in range(m+1):
        P.append(g(P[k]))
        e = np.abs(P[k+1] - P[k])
        p = P[k]

        if e < tol:
            break

    if k == m:
        return 'Maximum number of iterations was exceeded'

    return f'Root: {p}\nIteration: {k+1}\nAbsolute error: {e}\nSequence: {P[1:]}'

