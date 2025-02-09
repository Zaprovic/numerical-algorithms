import numpy as np
from ..diff.vector import rk4


def linear_shot(p, q, r, a, b, alpha, beta, n):
    h = (b - a) / n

    w = np.zeros(n + 1)
    t = np.linspace(a, b, n + 1)

    def F1(t, u):
        return np.array([u[1], q(t) * u[0] + p(t) * u[1] + r(t)])

    def F2(t, v):
        return np.array([v[1], q(t) * v[0] + p(t) * v[1]])

    yu = rk4(F1, a, b, [alpha, 0], n)
    yv = rk4(F2, a, b, [0, 1], n)

    u = yu[0, -1]
    v = yv[0, -1]

    for i in range(n + 1):
        w[i] = u[i] + (beta - u[n]) * v[i] / v[n]

    return t, w
