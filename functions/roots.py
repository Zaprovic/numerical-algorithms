import numpy as np
import pandas as pd
from typing import Callable, Tuple

np.set_printoptions(precision=4)


def _numerical_derivative(f: Callable[[float], float], x: float) -> float:
    h = 1e-6
    return (f(x + h) - f(x - h)) / (2 * h)


def _numerical_second_derivative(
    f: Callable[[float], float], x: float, h: float = 1e-6
) -> float:
    return (f(x + h) - 2 * f(x) + f(x - h)) / (h**2)


def bisect(
    f: Callable[[float], float], a: float, b: float, delta: float = 1e-6
) -> pd.DataFrame:
    """
    Find a root of function f using the bisection method.
        The bisection method is a root-finding algorithm that repeatedly halves an interval
        and then selects the subinterval in which a root must lie for further processing.

    Parameters
    ----------
    f : Callable[[float], float]
        The function for which to find a root.
    a : float
        The left endpoint of the initial interval.
    b : float
        The right endpoint of the initial interval.
    delta : float, optional
        The desired accuracy. Default is 1e-6.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the steps of the algorithm with columns:
        - "a": Left endpoint of the interval
        - "b": Right endpoint of the interval
        - "p": Midpoint of the interval
        - "f(p)": Function value at the midpoint
        - "Iteration": Iteration number

    Raises
    ------
    ValueError
        If f(a) * f(b) > 0, which means there's no guarantee of a root in [a, b].
        Or if the method fails to converge within the estimated number of iterations.

    Notes
    -----
    The algorithm terminates when either:
        - An exact root is found (f(p) = 0)
        - The interval size becomes smaller than the specified delta
        - The maximum number of iterations is reached
    """

    if f(a) * f(b) > 0:
        raise ValueError("There is no root in the interval")

    i = 0
    fa = f(a)
    n0 = int(np.floor(np.log2((b - a) / delta)))
    data = []

    while i <= n0:
        p = a + (b - a) / 2
        fp = f(p)

        data.append([a, b, p, fp, i + 1])

        if fp == 0 or (b - a) / 2 < delta:
            return pd.DataFrame(data, columns=["a", "b", "p", "f(p)", "Iteration"])

        i += 1

        if fa * fp > 0:
            a = p
            fa = fp
        else:
            b = p

    return ValueError(f"Method failed after {n0} iterations")


def fixedpt(
    g: Callable[[float], float], p0: float, tol: float = 1e-6, m: int = 100
) -> pd.DataFrame:
    """
    Fixed point iteration for finding roots of a function

    Parameters
    ----------
    g : Callable[[float], float]
        The function for which to find the root.
    p0 : float
        Initial approximation of the root.
    tol : float, optional
        Tolerance for the absolute error. Default value is 1e-6.
    m : int, optional
        Maximum number of iterations. Default value is 100.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the history of the iteration, with columns
        'Iteration', 'p', and 'Absolute Error'.

    Raises
    ------
    ValueError
        If the method fails after m iterations.
    """
    P = [p0]
    data = []

    for k in range(m + 1):
        P.append(g(P[k]))
        e = np.abs(P[k + 1] - P[k])
        p = P[k + 1]

        data.append([k + 1, p, e])

        if e < tol:
            return pd.DataFrame(
                data,
                columns=["Iteration", "p", "Absolute Error"],
            )

    raise ValueError("Maximum number of iterations was exceeded")


def newton(
    f: Callable[[float], float],
    x0: float,
    delta: float = 1e-6,
    epsilon: float = 1e-6,
    m: int = 100,
) -> Tuple[float, float, int, float]:
    """
    Finds a root of a function using the Newton-Raphson method.
    The method iteratively improves an initial approximation by:
    x_{n+1} = x_n - f(x_n)/f'(x_n)

    Parameters
    ----------
    f : Callable[[float], float]
        The function for which we want to find a root.
    x0 : float
        Initial approximation of the root.
    delta : float, optional
        Tolerance for the absolute difference between consecutive approximations.
        Default is 1e-6.
    epsilon : float, optional
        Tolerance for the function value at the approximation.
        Default is 1e-6.
    m : int, optional
        Maximum number of iterations. Default is 100.

    Returns
    -------
    Tuple[float, float, int, float]
        A tuple containing:
        - The approximation of the root
        - The absolute error estimate (|p - x0|)
        - The number of iterations performed
        - The function value at the approximation

    Raises
    ------
    ZeroDivisionError
        If the derivative at any point is zero or not defined.
    ValueError
        If the method fails to converge within the specified number of iterations.

    Notes
    -----
    The derivative is calculated numerically using the _numerical_derivative function.
    The method stops when either |f(p)| < epsilon or |p - x0| < delta.
    """
    for i in range(m + 1):
        df = _numerical_derivative(f, x0)

        if np.isnan(df):  # Avoid division by zero
            raise ZeroDivisionError("Division by zero")

        p = x0 - f(x0) / df
        e = np.abs(p - x0)
        x0 = p
        y = f(p)

        if (np.abs(y) < epsilon) or (e < delta):
            return p, e, i + 1, y

    raise ValueError("Method failed after {m} iterations")


def newtonMod(
    f: Callable[[float], float],
    x0: float,
    delta: float = 1e-6,
    epsilon: float = 1e-6,
    m: int = 100,
) -> Tuple[float, float, int, float]:
    """
    Modified Newton's method for finding roots of nonlinear equations.
    This method uses a modification to the standard Newton's method that includes
    second derivative information to accelerate convergence. It is particularly
    useful for multiple roots where the standard Newton's method might converge slowly.

    Parameters
    ----------
    f : Callable[[float], float]
        The function for which we want to find the root.
    x0 : float
        Initial guess for the root.
    delta : float
        Tolerance for the change in x between iterations. Default is 1e-6.
    epsilon : float
        Tolerance for the function value at the approximate root. Default is 1e-6.
    m : int
        Maximum number of iterations. Default is 100.

    Returns
    -------
    Tuple[float, float, int, float]
        - Approximation of the root
        - Absolute error estimate
        - Number of iterations performed
        - Function value at the approximated root

    Raises
    ------
    ZeroDivisionError
        If the denominator in the iteration formula becomes zero.
    ValueError
        If the method fails to converge within the specified number of iterations.

    Notes
    -----
    The formula used is:
        x_{n+1} = x_n - (f(x_n) * f'(x_n)) / (f'(x_n)^2 - f(x_n) * f''(x_n))
    This method generally has cubic convergence for simple roots and
    quadratic convergence for multiple roots, compared to the linear
    convergence of standard Newton's method for multiple roots.
    """
    for i in range(m + 1):
        df = _numerical_derivative(f, x0)
        d2f = _numerical_second_derivative(f, x0)

        denominator = df**2 - f(x0) * d2f
        if denominator == 0:  # Prevent division by zero
            raise ZeroDivisionError("Division by zero")

        p = x0 - (f(x0) * df) / denominator
        e = np.abs(p - x0)
        x0 = p
        y = f(p)

        if (np.abs(y) < epsilon) or (e < delta):
            return p, e, i + 1, y

    raise ValueError("Method failed after {m} iterations")
