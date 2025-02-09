import numpy as np
from typing import Callable, List, Tuple, Literal


def solve_ode_system(
    F: Callable[[float, np.ndarray], np.ndarray],
    a: float,
    b: float,
    alpha: list,
    n: int,
    method: Literal["euler", "eulermod", "rk4"],
) -> np.ndarray:
    """
    Generalized solver for systems of ODEs using different numerical methods.

    Parameters:
        F (Callable[[float, np.ndarray], np.ndarray]): Function defining the system dY/dt = F(t, Y).
        a (float): Initial time.
        b (float): Final time.
        alpha (list): Initial conditions for the system.
        n (int): Number of steps.
        method (str): Numerical method ("euler", "eulermod", "rk4").

    Returns:
        np.ndarray: Array of solutions [(t_values, Y_1), (t_values, Y_2), ...].
    """

    h = (b - a) / n
    t = np.linspace(a, b, n + 1)
    W = np.zeros((n + 1, len(alpha)))
    W[0] = alpha

    for i in range(n):
        if method == "euler":
            W[i + 1] = W[i] + h * np.array(F(t[i], W[i]))

        elif method == "eulermod":
            W[i + 1] = W[i] + (h / 2) * (
                np.array(F(t[i], W[i]))
                + np.array(F(t[i + 1], W[i] + h * np.array(F(t[i], W[i]))))
            )

        elif method == "rk4":
            k1 = h * np.array(F(t[i], W[i]))
            k2 = h * np.array(F(t[i] + h / 2, W[i] + k1 / 2))
            k3 = h * np.array(F(t[i] + h / 2, W[i] + k2 / 2))
            k4 = h * np.array(F(t[i + 1], W[i] + k3))

            W[i + 1] = W[i] + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        else:
            raise ValueError(f"Unknown method: {method}")

    W = W.T
    solutions = np.array([(t, W[i]) for i in range(len(W))])

    return solutions
