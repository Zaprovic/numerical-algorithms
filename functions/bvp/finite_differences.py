import matplotlib.pyplot as plt
import numpy as np
from typing import Callable


def finite_differences(
    p: Callable[[float], float],
    q: Callable[[float], float],
    r: Callable[[float], float],
    a: float,
    b: float,
    alpha: float,
    beta: float,
    n: int,
) -> np.ndarray:
    """
    Solves a boundary value problem for a second-order linear differential equation
    using the finite differences method.

    The differential equation is of the form:
        -(d^2u/dx^2) + p(x) * (du/dx) + q(x) * u = r(x),  a <= x <= b
    with boundary conditions:
        u(a) = alpha, u(b) = beta

    Parameters
    ----------
    p : Callable[[float], float]
        Function representing the coefficient of du/dx in the differential equation.
    q : Callable[[float], float]
        Function representing the coefficient of u in the differential equation.
    r : Callable[[float], float]
        Function representing the right-hand side of the differential equation.
    a : float
        Left boundary of the interval.
    b : float
        Right boundary of the interval.
    alpha : float
        Boundary condition at x = a.
    beta : float
        Boundary condition at x = b.
    n : int
        Number of subintervals for discretization.

    Returns
    -------
    np.ndarray
        Array containing the grid points (x) and the corresponding solution (y).
    """

    h = (b - a) / n

    x = np.linspace(a, b, n + 1)
    U = np.zeros((n - 1, n - 1))

    D = np.diag([-(2 / np.pow(h, 2) + q(x[i])) for i in range(1, n)])
    L = np.diag([1 / np.pow(h, 2) + (1 / (2 * h)) * p(x[i]) for i in range(2, n)], k=-1)
    U = np.diag(
        [1 / np.pow(h, 2) - (1 / (2 * h)) * p(x[i]) for i in range(1, n - 1)], k=1
    )

    # # Tridiagonal matrix
    M = D + L + U

    # vector b of the form Mx = b
    b = np.array([r(x[i]) for i in range(1, n)])
    b[0] = r(x[1]) - alpha * (1 / np.pow(h, 2) + p(x[1]) / (2 * h))
    b[-1] = r(x[n - 1]) - beta * (1 / np.pow(h, 2) - p(x[n - 1]) / (2 * h))

    # solution
    y = np.linalg.solve(M, b)
    y = np.concatenate([[alpha], y, [beta]])

    return np.array([x, y])
