import numpy as np


def lagrange(x: list[float], y: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the Lagrange polynomial interpolation and its coefficients.

    Parameters:
    -----------
    x : list[float]
        The x-coordinates of the data points.
    y : list[float]
        The y-coordinates of the data points.

    Returns:
    --------
    tuple[np.ndarray, np.ndarray]
    A tuple containing:
        - The coefficients of the interpolation polynomial in descending order
        - The Lagrange basis polynomials
    """

    n = len(x)
    y = np.array(y)

    # Calculating denominators
    d = np.prod([[x[i] - x[j] for j in range(n) if j != i] for i in range(n)], axis=1)

    # numerators
    T = np.array([[v for idx, v in enumerate(x) if idx != i] for i in range(n)])

    Tk = np.array([[[1, -i] for i in T[k]] for k in range(n)])

    Lk = []

    for idx, t in enumerate(Tk):
        r = [1]
        for seq in t:
            r = np.convolve(r, seq)
        Lk.append(r / d[idx])

    # Lagrange coefficients
    Lk = np.array(Lk)

    c = 0
    for i in range(n):
        c += Lk[i] * y[i]

    return c, Lk


def newtonpoly(x, y):
    """
    Computes the Newton form of the interpolating polynomial.

    This function calculates the coefficients of the Newton polynomial
    interpolation using divided differences method, which can be evaluated
    as p(x) = C[0] + C[1] * (x - x[0]) + C[2] * (x - x[0]) * (x - x[1]) + ...

    Parameters
    ----------
    x : array_like
        1-D array of x coordinates of the points.
    y : array_like
        1-D array of y coordinates of the points.

    Returns
    -------
    C : ndarray
        Coefficients of the Newton polynomial in descending order of powers.
        These are the coefficients of the Newton basis polynomials.
    F : ndarray
        The divided difference table. F[i,j] contains the j-th divided
        difference that involves the points x[i-j], x[i-j+1], ..., x[i].

    Notes
    -----
    The Newton polynomial is a form of polynomial interpolation that is
    particularly useful for adding new points to an existing interpolation.

    The algorithm computes the divided differences table and then
    converts these to the coefficients of the Newton basis polynomials.
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
