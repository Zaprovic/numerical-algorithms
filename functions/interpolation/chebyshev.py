import numpy as np


def chebyshevNodes(a, b, n, f):
    """
    Generates Chebyshev nodes in the interval [a, b] and computes
    the function f at those nodes.

    Parameters
    ----------
    a : float
        lower bound of the interval
    b : float
        upper bound of the interval
    n : int
        number of nodes
    f : function
        function to compute at the nodes

    Returns
    -------
    x : array_like
        Chebyshev nodes
    y : array_like
        f evaluated at the nodes
    """
    x = np.zeros(n + 1)

    for i in range(len((x))):
        x[i] = ((b - a) / 2) * np.cos((2 * i + 1) * np.pi / (2 * (n + 1))) + (a + b) / 2

    x.sort()
    y = f(x)

    return x, y
