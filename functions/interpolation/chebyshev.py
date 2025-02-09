import numpy as np

np.set_printoptions(suppress=True)


def helper():
    return """Chebyshev nodes
Function: chebyshevNodes(a,b,n,f)
f: Function (Can be declared as Python or lambda function)
a: Left side of the interval
b: Right side of the interval
n: Number of nodes"""


def chebyshevNodes(a, b, n, f):
    x = np.zeros(n + 1)

    for i in range(len((x))):
        x[i] = ((b - a) / 2) * np.cos((2 * i + 1) * np.pi / (2 * (n + 1))) + (a + b) / 2

    x.sort()
    y = f(x)

    return x, y
