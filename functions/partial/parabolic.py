from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


def forward(
    f: Callable[[float], float],
    alpha: float,
    g1: Callable[[float], float],
    g2: Callable[[float], float],
    xa: float,
    xb: float,
    ta: float,
    tb: float,
    h: float,
    k: float,
) -> np.ndarray:
    """
    Solves a hyperbolic partial differential equation using the forward method.

    Parameters
    ----------
    f : callable
        Initial condition function for u(x, t0).
    alpha : float
        Parameter related to the equation.
    g1 : callable
        Boundary condition function for u(xa, t).
    g2 : callable
        Boundary condition function for u(xb, t).
    xa : float
        Left boundary of the spatial interval.
    xb : float
        Right boundary of the spatial interval.
    ta : float
        Initial time.
    tb : float
        Final time.
    h : float
        Step size for the spatial variable x.
    k : float
        Step size for the time variable t.
    plot : bool, optional
        If not None, a 3D surface plot of the solution is displayed.

    Returns
    -------
    np.ndarray
        A 2D array representing the computed solution u(x, t) over the grid.
    """

    N = round((xb - xa) / h)
    M = round((tb - ta) / k)

    l = alpha**2 * k / h**2
    x = np.linspace(xa, xb, N + 1)
    t = np.linspace(ta, tb, M + 1)
    w = np.ones((len(t), len(x)))

    w[0] = f(x)
    w[1:, 0] = g1(t[1:])
    w[1:, -1] = g2(t[1:])

    for j in range(M):
        for i in range(1, N):
            w[j + 1, i] = l * (w[j, i + 1] - 2 * w[j, i] + w[j, i - 1]) + w[j, i]

    return w


def backward(
    f: Callable[[float], float],
    alpha: float,
    g1: Callable[[float], float],
    g2: Callable[[float], float],
    xa: float,
    xb: float,
    ta: float,
    tb: float,
    h: float,
    k: float,
    plot: bool = None,
) -> np.ndarray:
    """
    Solves a parabolic partial differential equation using the backward method.

    Parameters
    ----------
    f : callable
        Initial condition function for u(x, t0).
    alpha : float
        Parameter related to the equation.
    g1 : callable
        Boundary condition function for u(xa, t).
    g2 : callable
        Boundary condition function for u(xb, t).
    xa : float
        Left boundary of the spatial interval.
    xb : float
        Right boundary of the spatial interval.
    ta : float
        Initial time.
    tb : float
        Final time.
    h : float
        Step size for the spatial variable x.
    k : float
        Step size for the time variable t.
    plot : bool, optional
        If not None, a 3D surface plot of the solution is displayed.

    Returns
    -------
    np.ndarray
        A 2D array representing the computed solution u(x, t) over the grid.
    """

    N = round((xb - xa) / h)
    M = round((tb - ta) / k)

    l = alpha**2 * k / h**2
    x = np.linspace(xa, xb, N + 1)
    t = np.linspace(ta, tb, M + 1)
    w = np.zeros((len(t), len(x)))

    w[0] = f(x)
    w[1:, 0] = g1(t[1:])
    w[1:, -1] = g2(t[1:])

    # declare tridiagonal matrix
    J = diags([1 + 2 * l, -l, -l], [0, -1, 1], shape=(N - 1, N - 1)).tocsc()

    # vector to allocate all the right hand vectors of the system that needs to be solved for each i
    v = np.zeros((M, N - 1))

    # iterate over v to update the values in a loop
    vb = np.zeros(N - 1)
    for j in range(1, M + 1):
        vb.fill(0)
        vb[0] = w[j, 0]
        vb[-1] = w[j, N]
        v[j - 1] = w[j - 1, 1:N] + l * vb
        w[j, 1:-1] = spsolve(J, v[j - 1])

    return w


def crnich(
    f: Callable[[float], float],
    alpha: float,
    g1: Callable[[float], float],
    g2: Callable[[float], float],
    xa: float,
    xb: float,
    ta: float,
    tb: float,
    h: float,
    k: float,
    plot: bool = None,
) -> np.ndarray:
    """
    Solves a parabolic partial differential equation using the Crank-Nicholson method.

    Parameters
    ----------
    f : callable
        Initial condition function for u(x, t0).
    alpha : float
        Parameter related to the equation.
    g1 : callable
        Boundary condition function for u(xa, t).
    g2 : callable
        Boundary condition function for u(xb, t).
    xa : float
        Left boundary of the spatial interval.
    xb : float
        Right boundary of the spatial interval.
    ta : float
        Initial time.
    tb : float
        Final time.
    h : float
        Step size for the spatial variable x.
    k : float
        Step size for the time variable t.
    plot : bool, optional
        If not None, a 3D surface plot of the solution is displayed.

    Returns
    -------
    np.ndarray
        A 2D array representing the computed solution u(x, t) over the grid.
    """
    N = round((xb - xa) / h)
    M = round((tb - ta) / k)

    l = alpha**2 * k / h**2
    x = np.linspace(xa, xb, N + 1)
    t = np.linspace(ta, tb, M + 1)
    w = np.zeros((len(t), len(x)))

    w[0] = f(x)
    w[1:, 0] = g1(t[1:])
    w[1:, -1] = g2(t[1:])

    # optimized way to declare tri-diagonal matrix using scipy sparse module
    T = diags([1 + l, -l / 2, -l / 2], [0, -1, 1], shape=(N - 1, N - 1)).tocsc()

    # vector to allocate all the right hand vectors of the system that needs to be solved for each i
    v = np.zeros((M, N - 1))

    # iterate over v to update the values in a loop
    for j in range(M):
        vb = np.zeros(N - 1)
        vb[0] = w[j + 1, 0] + w[j, 0]
        vb[-1] = w[j + 1, N] + w[j, N]

        J = diags([1 - l, l / 2, l / 2], [0, -1, 1], shape=(N - 1, N - 1)).tocsc()
        v[j] = J.dot(w[j, 1:-1]) + (l / 2) * vb
        w[j + 1, 1:-1] = spsolve(T, v[j])

    if plot is not None:
        X, T = np.meshgrid(x, t)
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X, T, w, cmap="plasma")
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        ax.set_zlabel("u(x, t)")
        plt.show()

    return w
