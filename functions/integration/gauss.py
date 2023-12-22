import numpy as np
from scipy.special import roots_legendre


def gauss(f, a, b, n):
    """
    Parameters
    ----------
    f
    a
    b
    n

    Returns
    -------

    """

    x, c = roots_legendre(n)
    F = [f((b-a)*i/2 + (b+a)/2) for i in x]
    F = np.multiply(c, F)

    return (b-a)*sum(F)/2
