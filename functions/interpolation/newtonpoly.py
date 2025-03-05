import matplotlib.pyplot as plt
import numpy as np


def newtonpoly(x, y):
    """
    Constructs the Newton polynomial and its divided differences table for a given set of points.

    Parameters
    ----------
    x : array_like
        x-coordinates of the data points.
    y : array_like
        y-coordinates of the data points.

    Returns
    -------
    C : array_like
        Coefficients of the Newton polynomial in descending order of degree.
    F : ndarray
        Divided differences table. The first column represents the y values, and
        subsequent columns represent the divided differences.
    """

    n = len(x)
    F = np.zeros((n, n))
    F[:, 0] = y

    for i in range(1, n):
        for j in range(1, i + 1):
            F[i, j] = (F[i, j - 1] - F[i - 1, j - 1]) / (x[i] - x[i - j])

    C = [F[n - 1, n - 1]]

    for k in range(n - 2, -1, -1):
        C = np.convolve(C, [1, -x[k]])  # Multiply by (x - X[k])
        C[-1] += F[k, k]  # Add divided difference term

    return C, F
