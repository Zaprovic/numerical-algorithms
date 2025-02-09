import numpy as np
from typing import Callable, List, Tuple


def euler(
    F: Callable[[float, np.ndarray], np.ndarray],
    a: float,
    b: float,
    alpha: list,
    n: int,
) -> np.array:

    h = (b - a) / n

    t = np.linspace(a, b, n + 1)
    W = np.zeros((n + 1, len(alpha)))
    W[0] = alpha

    for i in range(n):
        W[i + 1] = W[i] + h * np.array(F(t[i], W[i]))

    W = W.T

    solutions = np.array([(t, W[i]) for i in range(len(W))])

    return solutions


def eulermod(F: callable, a: float, b: float, alpha: list, n: int) -> np.array:
    h = (b - a) / n

    t = np.linspace(a, b, n + 1)
    W = np.zeros((n + 1, len(alpha)))
    W[0] = alpha

    for i in range(n):
        W[i + 1] = W[i] + (h / 2) * (
            np.array(F(t[i], W[i]))
            + np.array(F(t[i + 1], W[i] + h * np.array(F(t[i], W[i]))))
        )

    W = W.T

    solutions = np.array([(t, W[i]) for i in range(len(W))])

    return solutions


def rk4(F: callable, a: float, b: float, alpha: list, n: int) -> np.array:
    h = (b - a) / n

    t = np.linspace(a, b, n + 1)
    W = np.zeros((n + 1, len(alpha)))
    W[0] = alpha

    for i in range(n):
        k1 = h * np.array(F(t[i], W[i]))
        k2 = h * np.array(F(t[i] + h / 2, W[i] + k1 / 2))
        k3 = h * np.array(F(t[i] + h / 2, W[i] + k2 / 2))
        k4 = h * np.array(F(t[i + 1], W[i] + k3))

        W[i + 1] = W[i] + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    W = W.T

    solutions = np.array([(t, W[i]) for i in range(len(W))])

    return solutions
