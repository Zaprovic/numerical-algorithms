import numpy as np


def trapezoidal(f, a, b, n):
    h = (b - a) / n

    x = np.array([a + i * (b - a) / n for i in range(n + 1)])
    y = f(x)

    S = (h / 2) * (y[0] + 2 * np.sum([y[i] for i in range(1, n)]) + y[-1])

    return S


def trapezoidal_discrete(y, a, b, n):
    h = (b - a) / n
    S = (h / 2) * (y[0] + 2 * np.sum([y[i] for i in range(1, n)]) + y[-1])

    return S


def simpson(f, a, b, n):
    h = (b - a) / n

    x = np.array([a + i * (b - a) / n for i in range(n + 1)])
    y = f(x)

    S = (h / 3) * (
        y[0]
        + 4 * np.sum([y[2 * i - 1] for i in range(1, int(n / 2 + 1))])
        + 2 * np.sum([y[2 * i] for i in range(1, int(n / 2 - 1))])
        + y[-1]
    )
    return S


def simpson_discrete(y, a, b, n):
    h = (b - a) / n
    S = (h / 3) * (
        y[0]
        + 4 * sum([y[2 * i - 1] for i in range(1, int(n / 2 + 1))])
        + 2 * sum([y[2 * i] for i in range(1, int(n / 2 - 1))])
        + y[-1]
    )

    return S
