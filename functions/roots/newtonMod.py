from numpy import abs
from scipy.misc import derivative


def helper():
    return '''Newton's modified method for transcendental equations
    
Function : newtonMod(f, df, d2f, x0, tol)
f : Function must be created as a Python function or using lambdas
x0 : Initial approximation of the root
tol : Maximum desired tolerance for the output
'''


def newtonMod(f, x0, delta, epsilon, m):

    def df(x): return derivative(f, x, dx=1e-6)
    def d2f(x): return derivative(df, x, dx=1e-6)

    for i in range(m + 1):
        p = x0 - (f(x0) * df(x0)) / (df(x0) ** 2 - f(x0) * d2f(x0))
        e = abs(p - x0)
        x0 = p
        y = f(p)

        if (abs(y) < epsilon) or (e < delta):
            break

        return f'Root: {p}\nAbsolute error: {e} \nIteration: {i + 1}\nf(root): {y}'
