import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from typing import Callable, Tuple

# y' = f(t,y)
# a <= t <= b,  y(a) = alpha


def euler(
    f: Callable[[float, float], float], a: float, b: float, alpha: float, n: int
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Implements Euler's method for solving ODEs.

    Parameters:
        f (Callable[[float, float], float]): The function f(t, y) representing dy/dt.
        a (float): The initial time value.
        b (float): The final time value.
        alpha (float): The initial condition y(a).
        n (int): The number of steps.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Arrays of time values and approximate solutions.
    """

    h: float = (b - a) / n
    t: npt.NDArray[np.float64] = np.linspace(a, b, n + 1, dtype=np.float64)
    w: npt.NDArray[np.float64] = np.zeros(n + 1, dtype=np.float64)
    w[0] = alpha

    for i in range(n):
        w[i + 1] = w[i] + h * f(t[i], w[i])

    return t, w


def eulermod(
    f: Callable[[float, float], float], a: float, b: float, alpha: float, n: int
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Implements Euler's modified method for solving ODEs.

    Parameters:
        f (Callable[[float, float], float]): The function f(t, y) representing dy/dt.
        a (float): The initial time value.
        b (float): The final time value.
        alpha (float): The initial condition y(a).
        n (int): The number of steps.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Arrays of time values and approximate solutions.
    """

    h: float = (b - a) / n

    t: np.ndarray = np.linspace(a, b, n + 1, dtype=np.float64)
    w: np.ndarray = np.zeros(n + 1, dtype=np.float64)
    w[0] = alpha

    for i in range(n):
        w[i + 1] = w[i] + (h / 2) * (
            f(t[i], w[i]) + f(t[i + 1], w[i] + h * f(t[i], w[i]))
        )

    return t, w


def heun(
    f: Callable[[float, float], float], a: float, b: float, alpha: float, n: int
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Implements Heun's method for solving ODEs.

    Parameters:
        f (Callable[[float, float], float]): The function f(t, y) representing dy/dt.
        a (float): The initial time value.
        b (float): The final time value.
        alpha (float): The initial condition y(a).
        n (int): The number of steps.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Arrays of time values and approximate solutions.
    """

    h: float = (b - a) / n

    t: np.ndarray = np.linspace(a, b, n + 1, dtype=np.float64)
    w: np.ndarray = np.zeros(n + 1, dtype=np.float64)
    w[0] = alpha

    for i in range(n):
        w[i + 1] = w[i] + (h / 4) * (
            f(t[i], w[i])
            + 3
            * f(
                t[i] + 2 * h / 3,
                w[i] + (2 * h / 3) * f(t[i] + h / 3, w[i] + (h / 3) * f(t[i], w[i])),
            )
        )
    return t, w


def midpoint(
    f: Callable[[float, float], float], a: float, b: float, alpha: float, n: int
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Implements Midpoint method for solving ODEs.

    Parameters:
        f (Callable[[float, float], float]): The function f(t, y) representing dy/dt.
        a (float): The initial time value.
        b (float): The final time value.
        alpha (float): The initial condition y(a).
        n (int): The number of steps.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Arrays of time values and approximate solutions.
    """

    h: float = (b - a) / n

    t: np.ndarray = np.linspace(a, b, n + 1, dtype=np.float64)
    w: np.ndarray = np.zeros(n + 1, dtype=np.float64)
    w[0] = alpha

    for i in range(n):
        w[i + 1] = w[i] + h * f(t[i] + h / 2, w[i] + (h / 2) * f(t[i], w[i]))

    return t, w


def rk4(
    f: Callable[[float, float], float], a: float, b: float, alpha: float, n: int
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Implements RK4 method for solving ODEs.

    Parameters:
        f (Callable[[float, float], float]): The function f(t, y) representing dy/dt.
        a (float): The initial time value.
        b (float): The final time value.
        alpha (float): The initial condition y(a).
        n (int): The number of steps.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Arrays of time values and approximate solutions.
    """

    h: float = (b - a) / n

    t: np.ndarray = np.linspace(a, b, n + 1, dtype=np.float64)
    w: np.ndarray = np.zeros(n + 1, dtype=np.float64)
    w[0] = alpha

    for i in range(n):
        k1 = h * f(t[i], w[i])
        k2 = h * f(t[i] + h / 2, w[i] + k1 / 2)
        k3 = h * f(t[i] + h / 2, w[i] + k2 / 2)
        k4 = h * f(t[i + 1], w[i] + k3)

        w[i + 1] = w[i] + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return t, w
