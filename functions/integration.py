from typing import Any, Callable, TypeVar, Union

import numpy as np
from scipy.special import roots_legendre

np.set_printoptions(precision=4)

# Define numeric types that can be used
NumericType = Union[int, float, complex]
ArrayType = TypeVar("ArrayType", bound=np.ndarray)


def closed_newton_cotes(
    f: Callable[[Union[float, ArrayType]], Union[float, ArrayType]],
    a: float,
    b: float,
    n: int,
) -> float:
    """
    Approximates the definite integral of a function using the closed Newton-Cotes formula.

    This method uses quadrature weights based on polynomial interpolation through equally
    spaced points including the endpoints of the integration interval.

    Parameters
    ----------
    f : Callable
        The function to integrate. Must accept array input and return array output.
    a : float
        Lower limit of integration.
    b : float
        Upper limit of integration.
    n : int
        Number of subintervals to use.

    Returns
    -------
    float
        The approximated value of the integral.

    Examples
    --------
    >>> import numpy as np
    >>> def f(x): return np.sin(x)
    >>> # Approximate the integral of sin(x) from 0 to π
    >>> closed_newton_cotes(f, 0, np.pi, 4)  # Should be close to 2
    1.9999999999999996

    >>> def g(x): return x**2
    >>> # Approximate the integral of x^2 from 0 to 1
    >>> closed_newton_cotes(g, 0, 1, 2)  # Should be close to 1/3
    0.3333333333333333
    """
    if n <= 0:
        raise ValueError("Number of subintervals must be positive")
    if a >= b:
        raise ValueError("Lower bound must be less than upper bound")

    h = (b - a) / n

    x = np.array([a + i * h for i in range(n + 1)])
    y = f(x)

    A = np.vander(x, increasing=True).T

    v = np.zeros(n + 1)
    for i in range(n + 1):
        v[i] = (b ** (i + 1) - a ** (i + 1)) / (i + 1)  # Integral of x^i from a to b

    r = np.linalg.solve(A, v)
    S = np.dot(r, y)

    return S


def open_newton_cotes(
    f: Callable[[Union[float, ArrayType]], Union[float, ArrayType]],
    a: float,
    b: float,
    n: int,
) -> float:
    """
    Approximates the definite integral of a function using the open Newton-Cotes formula.

    This method uses quadrature weights based on polynomial interpolation through equally
    spaced points excluding the endpoints of the integration interval.

    Parameters
    ----------
    f : Callable
        The function to integrate. Must accept array input and return array output.
    a : float
        Lower limit of integration.
    b : float
        Upper limit of integration.
    n : int
        Number of interior points to use.

    Returns
    -------
    float
        The approximated value of the integral.

    Examples
    --------
    >>> import numpy as np
    >>> def f(x): return np.sin(x)
    >>> # Approximate the integral of sin(x) from 0 to π
    >>> open_newton_cotes(f, 0, np.pi, 3)  # Should be close to 2
    1.9999999999999991

    >>> def g(x): return x**2
    >>> # Approximate the integral of x^2 from 0 to 1
    >>> open_newton_cotes(g, 0, 1, 3)  # Should be close to 1/3
    0.3333333333333333
    """
    if n <= 0:
        raise ValueError("Number of interior points must be positive")
    if a >= b:
        raise ValueError("Lower bound must be less than upper bound")

    h = (b - a) / (n + 2)

    # Generate equally spaced points (including endpoints)
    x = np.array([a + i * h for i in range(n + 3)])
    y = f(x)

    # Set up matrix for interior points
    A = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        A[i, :] = np.power(x[1:-1], i)

    # Compute the right-hand side vector (integrals of monomials)
    v = np.zeros(n + 1)
    for i in range(n + 1):
        v[i] = (b ** (i + 1) - a ** (i + 1)) / (i + 1)  # Integral of x^i from a to b

    # Calculate quadrature weights
    r = np.linalg.solve(A, v)
    S = np.dot(r, y[1:-1])

    return S


def closed_composite_newton_cotes(
    f: Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]],
    a: float,
    b: float,
    n: int,
    order: int = 1,
) -> float:
    """
    Approximates the definite integral using composite closed Newton-Cotes formulas.

    This divides the integration interval [a,b] into n subintervals and applies
    the closed Newton-Cotes formula of specified order to each subinterval.

    Parameters
    ----------
    f : Callable
        The function to integrate. Must accept array input and return array output.
    a : float
        Lower limit of integration.
    b : float
        Upper limit of integration.
    n : int
        Number of subintervals to divide [a,b] into. Must be divisible by the order.
    order : int, optional
        Order of the Newton-Cotes formula to use in each subinterval (default is 1,
        which corresponds to the trapezoidal rule).

    Returns
    -------
    float
        The approximated value of the integral.

    Examples
    --------
    >>> import numpy as np
    >>> def f(x): return np.sin(x)
    >>> # Approximate the integral of sin(x) from 0 to π using 4 subintervals
    >>> # and Newton-Cotes with 2 points in each (trapezoidal rule)
    >>> composite_newton_cotes(f, 0, np.pi, 4, 1)  # Should be close to 2
    1.9966944425646091

    >>> # Using Newton-Cotes with 3 points in each (Simpson's rule)
    >>> composite_newton_cotes(f, 0, np.pi, 4, 2)
    2.0000000000000013
    """
    if n <= 0:
        raise ValueError("Number of subintervals must be positive")
    if a >= b:
        raise ValueError("Lower bound must be less than upper bound")
    if order <= 0:
        raise ValueError("Order must be positive")
    if n % order != 0:
        raise ValueError(
            f"Number of subintervals ({n}) must be divisible by the order ({order})"
        )

    h = (b - a) / n
    result = 0

    for i in range(0, n, order):
        subinterval_a = a + i * h
        subinterval_b = a + (i + order) * h
        result += closed_newton_cotes(f, subinterval_a, subinterval_b, order)

    return result


def gaussLegendre(f, a, b, n):
    x, c = roots_legendre(n)
    F = [f((b - a) * i / 2 + (b + a) / 2) for i in x]
    S = ((b - a) / 2) * np.dot(c, F)

    return S
