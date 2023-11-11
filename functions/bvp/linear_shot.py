import matplotlib.pyplot as plt
import numpy as np


def linear_shot(p, q, r, a, b, alpha, beta, n, plot=None):
    """
    Parameters: problem in the form of y" = p(x)y' + q(x)y + r(x)
    ----------
    p
    q
    r
    a
    b
    alpha
    beta
    n
    plot

    Returns
    -------

    """
    h = (b - a) / n

    # Initializing array of values for the solution y
    w = np.zeros(n + 1)

    # First initial value problem y1
    u = np.zeros((n + 1, n + 1))
    u[0, 0] = alpha
    t = np.linspace(a, b, n + 1)

    def F1(t, u): return np.array([
        u[1],
        q(t) * u[0] + p(t) * u[1] + r(t),
        *[0] * (n - 1)
    ])

    # Second initial value problem y2
    v = np.zeros((n + 1, n + 1))
    v[0, 1] = 1

    def F2(t, v): return np.array([
        v[1],
        q(t) * v[0] + p(t) * v[1],
        *[0] * (n - 1)

    ])

    for i in range(n):
        ku1 = np.array(F1(t[i], u[i, :]))
        ku2 = np.array(F1(t[i] + h / 2, u[i, :] + ku1 * h / 2))
        ku3 = np.array(F1(t[i] + h / 2, u[i, :] + ku2 * h / 2))
        ku4 = np.array(F1(t[i + 1], u[i] + ku3 * h))

        kv1 = np.array(F2(t[i], v[i, :]))
        kv2 = np.array(F2(t[i] + h / 2, v[i, :] + kv1 * h / 2))
        kv3 = np.array(F2(t[i] + h / 2, v[i, :] + kv2 * h / 2))
        kv4 = np.array(F2(t[i + 1], v[i] + kv3 * h))

        u[i + 1, :] = u[i, :] + (h / 6) * (ku1 + 2 * ku2 + 2 * ku3 + ku4)
        v[i + 1, :] = v[i, :] + (h / 6) * (kv1 + 2 * kv2 + 2 * kv3 + kv4)

    y1 = u[:, 0]
    y2 = v[:, 0]

    for i in range(n + 1):
        w[i] = y1[i] + (beta - y1[n]) * y2[i] / y2[n]

    if plot is not None:
        plt.style.use("fivethirtyeight")
        plt.figure()
        plt.scatter(t, w)
        plt.show()
        return w

    elif plot == "continuous":
        plt.style.use("fivethirtyeight")
        plt.figure()
        plt.plot(t, w)
        plt.show()
        return w

    return w
