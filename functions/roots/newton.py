import numpy as np
from typing import Callable, Tuple


def _numerical_derivative(f: Callable[[float], float], x: float) -> float:
    h = 1e-6
    return (f(x + h) - f(x - h)) / (2 * h)


def _numerical_second_derivative(
    f: Callable[[float], float], x: float, h: float = 1e-6
) -> float:
    return (f(x + h) - 2 * f(x) + f(x - h)) / (h**2)


def newton(
    f: Callable[[float], float],
    x0: float,
    delta: float = 1e-6,
    epsilon: float = 1e-6,
    m: int = 100,
) -> Tuple[float, float, int, float] | str:
    """
    Newton's method for finding roots of a function.

    Parameters
    ----------
    f : Callable[[float], float]
        The function for which to find the root.
    x0 : float
        Initial guess for the root.
    delta : float
        Tolerance for the change in x between iterations.
    epsilon : float
        Tolerance for the value of the function at the root.
    m : int
        Maximum number of iterations.

    Returns
    -------
    tuple
        A tuple containing the root, the absolute error, the number of iterations,
        and the function value at the root, or a string message if the method fails.
    """

    for i in range(m + 1):
        df = _numerical_derivative(f, x0)

        if df == 0:  # Avoid division by zero
            return "Derivative is zero, Newton's method fails."

        p = x0 - f(x0) / df
        e = np.abs(p - x0)
        x0 = p
        y = f(p)

        if (np.abs(y) < epsilon) or (e < delta):
            return p, e, i + 1, y

    return "Max number of iterations exceeded"


def newtonMod(
    f: Callable[[float], float], x0: float, delta: float, epsilon: float, m: int
) -> Tuple[float, float, int, float] | str:
    """
    Modified Newton's method for finding roots of a function.

    Parameters
    ----------
    f : Callable[[float], float]
        The function for which to find the root.
    x0 : float
        Initial guess for the root.
    delta : float
        Tolerance for the change in x between iterations.
    epsilon : float
        Tolerance for the value of the function at the root.
    m : int
        Maximum number of iterations.

    Returns
    -------
    Tuple[float, float, int, float]
        A tuple containing the root, the absolute error, the number of iterations,
        and the function value at the root, or a string message if the method fails.
    """

    for i in range(m + 1):
        df = _numerical_derivative(f, x0)
        d2f = _numerical_second_derivative(f, x0)

        denominator = df**2 - f(x0) * d2f
        if denominator == 0:  # Prevent division by zero
            return "Newton's method fails: denominator is zero."

        p = x0 - (f(x0) * df) / denominator
        e = np.abs(p - x0)
        x0 = p
        y = f(p)

        if (np.abs(y) < epsilon) or (e < delta):
            return p, e, i + 1, y

    return "Max number of iterations exceeded"
