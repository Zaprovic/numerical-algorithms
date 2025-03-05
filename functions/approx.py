import numpy as np


def lspoly(x, y, n):
    """
    Computes the coefficients of the least-squares polynomial approximation of degree n.

    This function calculates the coefficients of a polynomial p(x) = a_0 * x^n + a_1 * x^(n-1) + ... + a_n
    that minimizes the sum of squared errors between the polynomial and the given data points.

    Parameters
    ----------
    x : array_like
        x-coordinates of the data points
    y : array_like
        y-coordinates of the data points
    n : int
        Degree of the polynomial approximation

    Returns
    -------
    ndarray
        Coefficients of the least-squares polynomial in descending order of powers
        (i.e., [a_0, a_1, ..., a_n])

    Notes
    -----
    The implementation solves a system of normal equations to find the coefficients
    that minimize the sum of squared errors.

    Requires NumPy for matrix operations.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([0, 1, 2, 3, 4])
    >>> y = np.array([1, 3, 7, 13, 21])
    >>> lspoly(x, y, 2)
    array([1., 1., 1.])  # Represents y = 1*x^2 + 1*x + 1
    """
    m = len(x)  # number of points in the discrete approximation

    A = np.zeros((n + 1, n + 1))

    for i in range(n + 1):
        for j in range(n + 1):
            A[i, j] = sum(x[k] ** (2 * n - i - j) for k in range(m))

    B = [sum([x[j] ** (n - i) * y[j] for j in range(m)]) for i in range(n + 1)]

    r = np.linalg.solve(A, B)

    return r
