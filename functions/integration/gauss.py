import numpy as np
from scipy.special import roots_legendre


def gauss(f, a, b, n):
    x, c = roots_legendre(n)
    F = [f((b-a)*i/2 + (b+a)/2) for i in x]
    S = ((b-a)/2) * np.dot(c,F)

    return S


