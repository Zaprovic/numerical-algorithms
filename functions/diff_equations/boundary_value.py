from typing import Callable, Tuple

import numpy as np

from .pvi import vector_ode as sd


def linear_shot(
    p: Callable[[float], float],
    q: Callable[[float], float],
    r: Callable[[float], float],
    a: float,
    b: float,
    alpha: float,
    beta: float,
    n: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solves a second-order linear boundary value problem using the shooting method.

    The boundary value problem is of the form:
    y'' + p(t)y' + q(t)y = r(t)
    with boundary conditions y(a) = alpha, y(b) = beta

    Parameters
    ----------
    p : callable[[float], float]
        Function p(t) in the differential equation.
    q : callable[[float], float]
        Function q(t) in the differential equation.
    r : callable[[float], float]
        Function r(t) in the differential equation.
    a : float
        Left boundary point.
    b : float
        Right boundary point.
    alpha : float
        Boundary condition at a (y(a) = alpha).
    beta : float
        Boundary condition at b (y(b) = beta).
    n : int
        Number of intervals to divide [a, b] into.

    Returns
    -------
    t : np.ndarray
        Array of mesh points from a to b.
    w : np.ndarray
        Approximation to the solution y at the mesh points.

    Notes
    -----
    The method uses Runge-Kutta 4th order (rk4) method for solving the initial value problems.

    Example
    -------
    >>> import numpy as np
    >>> from functions.boundary_value import linear_shot
    >>> from scipy.special import j0  # Bessel function of the first kind
    >>>
    >>> # Example: y'' + (1/t)y' + y = 0 with boundary conditions y(1) = j0(1), y(2) = j0(2)
    >>> # This has the exact solution y(t) = j0(t)
    >>> def p(t): return 1/t
    >>> def q(t): return 1
    >>> def r(t): return 0
    >>>
    >>> a, b = 1, 2
    >>> alpha, beta = j0(1), j0(2)
    >>> n = 100
    >>>
    >>> t, w = linear_shot(p, q, r, a, b, alpha, beta, n)
    >>>
    >>> # Compare with exact solution
    >>> exact = j0(t)
    >>> max_error = np.max(np.abs(w - exact))
    >>> print(f"Maximum error: {max_error:.6e}")
    """
    h = (b - a) / n

    w = np.zeros(n + 1)
    t = np.linspace(a, b, n + 1)

    def F1(t, u):
        return np.array([u[1], q(t) * u[0] + p(t) * u[1] + r(t)])

    def F2(t, v):
        return np.array([v[1], q(t) * v[0] + p(t) * v[1]])

    _, yu = sd(F1, a, b, [alpha, 0], n, "rk4")
    _, yv = sd(F2, a, b, [0, 1], n, "rk4")

    u = yu[:, 0]
    v = yv[:, 0]

    for i in range(n + 1):
        w[i] = u[i] + (beta - u[n]) * v[i] / v[n]

    return t, w


def finite_differences(
    p: Callable[[float], float],
    q: Callable[[float], float],
    r: Callable[[float], float],
    a: float,
    b: float,
    alpha: float,
    beta: float,
    n: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solves a second-order linear boundary value problem using the finite difference method.

    The boundary value problem is of the form:
    y'' + p(t)y' + q(t)y = r(t)
    with boundary conditions y(a) = alpha, y(b) = beta

    Parameters
    ----------
    p : Callable[[float], float]
        Function p(t) representing the coefficient of y' in the differential equation.
    q : Callable[[float], float]
        Function q(t) representing the coefficient of y in the differential equation.
    r : Callable[[float], float]
        Function r(t) representing the right-hand side of the differential equation.
    a : float
        Left boundary point.
    b : float
        Right boundary point.
    alpha : float
        Boundary condition at a (y(a) = alpha).
    beta : float
        Boundary condition at b (y(b) = beta).
    n : int
        Number of intervals to divide [a, b] into.

    Returns
    -------
    x : np.ndarray
        Array of mesh points from a to b.
    y : np.ndarray
        Approximation to the solution at the mesh points.

    Notes
    -----
    The method uses the finite difference approximation to convert the boundary value problem
    into a system of linear equations, which is then solved directly.

    Example
    -------
    >>> import numpy as np
    >>> from functions.boundary_value import finite_differences
    >>> from scipy.special import j0  # Bessel function of the first kind
    >>>
    >>> # Example: y'' + (1/t)y' + y = 0 with boundary conditions y(1) = j0(1), y(2) = j0(2)
    >>> # This has the exact solution y(t) = j0(t)
    >>> def p(t): return 1/t
    >>> def q(t): return 1
    >>> def r(t): return 0
    >>>
    >>> a, b = 1, 2
    >>> alpha, beta = j0(1), j0(2)
    >>> n = 100
    >>>
    >>> x, y = finite_differences(p, q, r, a, b, alpha, beta, n)
    >>>
    >>> # Compare with exact solution
    >>> exact = j0(x)
    >>> max_error = np.max(np.abs(y - exact))
    >>> print(f"Maximum error: {max_error:.6e}")
    """
    h = (b - a) / n

    x = np.linspace(a, b, n + 1)

    D = np.diag([-(2 / h**2 + q(x[i])) for i in range(1, n)])
    L = np.diag([1 / h**2 + (1 / (2 * h)) * p(x[i]) for i in range(2, n)], k=-1)
    U = np.diag([1 / h**2 - (1 / (2 * h)) * p(x[i]) for i in range(1, n - 1)], k=1)

    # # Tridiagonal matrix
    M = D + L + U

    # vector b of the form Mx = b
    b = np.array([r(x[i]) for i in range(1, n)])
    b[0] -= alpha * (1 / h**2 + p(x[1]) / (2 * h))
    b[-1] -= beta * (1 / h**2 - p(x[n - 1]) / (2 * h))

    # solution
    y = np.linalg.solve(M, b)
    y = np.concatenate([[alpha], y, [beta]])

    return x, y
