import numpy as np
from typing import Union


def lspoly(x: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
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
    # Input validation
    if n < 0:
        raise ValueError("Polynomial degree must be non-negative")
    if len(x) != len(y):
        raise ValueError("Input arrays x and y must have the same length")

    # this is to avoid poorly conditioned matrix
    if len(x) <= n:
        raise ValueError(
            f"Number of data points ({len(x)}) must be greater than polynomial degree ({n}). This is to avoid poorly conditioned matrix."
        )

    # Convert inputs to numpy arrays if they aren't already
    x = np.asarray(x)
    y = np.asarray(y)

    # Method 1: Using NumPy's polyfit (recommended approach)
    # Note: polyfit returns coefficients in decreasing power order, which matches our docstring
    coeffs = np.polynomial.Polynomial.fit(x, y, n)
    return coeffs.convert().coef[::-1]


def eval_poly(
    coeffs: np.ndarray, x: Union[np.ndarray, float]
) -> Union[np.ndarray, float]:
    """
    Evaluates a polynomial with given coefficients at point(s) x.

    Parameters
    ----------
    coeffs : array_like
        Coefficients of the polynomial in descending order of powers
    x : array_like or scalar
        Points at which to evaluate the polynomial

    Returns
    -------
    scalar or ndarray
        Value(s) of the polynomial at x
    """
    return np.polyval(coeffs, x)
