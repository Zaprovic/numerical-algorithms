import matplotlib.pyplot as plt
import numpy as np


def lspoly(x, y, n):
    """
    Parameters
    ----------
    x : x coordinates
    y : y coordinates
    n : Degree of polynomial approximation
    """
    m = len(x)  # number of points in the discrete approximation

    A = np.zeros((n+1, n+1))

    for i in range(n+1):
        for j in range(n+1):
            A[i, j] = sum(x[k]**(2*n - i - j) for k in range(m))

    B = [sum([x[j]**(n-i) * y[j] for j in range(m)]) for i in range(n+1)]

    r = np.linalg.solve(A, B)

    return r


def plot(x, y, P):

    xx = np.linspace(x[0], x[-1], 1500)
    yy = [np.polyval(P, xx[i]) for i in range(len(xx))]

    plt.grid()
    plt.scatter(x, y, c="k")
    plt.plot(xx, yy, linewidth=1.5)

    return plt.show()


x = [-3, -1, 2, 3, 7]
y = [5, 4, 12, 6, 0]

x = np.linspace(0.1, 1, 20)
y = [x[k] + np.cos(np.sqrt(k+1)) for k in range(len(x))]
n = 3
L = lspoly(x, y, n)
c = plot(x, y, L)
