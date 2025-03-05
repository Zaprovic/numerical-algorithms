import numpy as np


def lagrange(x, y):
    """
    Compute the Lagrange polynomial of degree n-1 passing through the given n points.

    Parameters
    ----------
    x : array_like
        x-coordinates of the points.
    y : array_like
        y-coordinates of the points.

    Returns
    -------
    c : array_like
        Coefficients of the Lagrange polynomial in descending order of degree.
    Lk : array_like
        Lagrange coefficients of each polynomial term, in descending order of degree.

    Notes
    -----
    The Lagrange polynomial is computed using the formula

    .. math::
        L(x) = \\sum_{i=0}^{n-1} y_i \\prod_{j=0,j \\neq i}^{n-1} \\frac{x - x_j}{x_i - x_j}

    which is a sum of products of (x - x_j) terms, where j is any index except i.

    The coefficients c are computed by multiplying the Lagrange coefficients Lk by the y values.

    The Lagrange coefficients Lk are computed by dividing the numerators by the denominators.
    The numerators are computed by multiplying the x-coordinates together, and the denominators are computed by taking the product of the differences between the x-coordinates and x[i].

    The returned coefficients c are in descending order of degree, i.e. c[0] is the coefficient of the highest degree term.
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
