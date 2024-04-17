import matplotlib.pyplot as plt
import numpy as np


def euler(f, a, b, alpha, n):
    """
    Parameters
    ----------
    f
    a
    b
    alpha: Initial condition
    n

    Returns
    -------

    """
    h = (b-a) / n

    t = np.linspace(a, b, n+1)
    w = np.zeros(n+1)
    w[0] = alpha

    for i in range(n):
        w[i+1] = w[i] + h*f(t[i], w[i])

    plt.grid()
    if h >= 0.1:
        plt.scatter(t, w, c='k', marker='.')
        plt.plot(t, w)
        plt.show()

    else:
        plt.plot(t, w)
        plt.show()

    return w

def eulermod(f, a, b, alpha, n):
    h = (b-a) / n

    t = np.linspace(a, b, n+1)
    w = np.zeros(n+1)
    w[0] = alpha

    for i in range(n):
        w[i+1] = w[i] + (h/2) * (f(t[i], w[i]) +
                                 f(t[i+1], w[i] + h*f(t[i], w[i])))

    plt.grid()
    if h > 0.1:
        plt.scatter(t, w, c='r')
        plt.plot(t, w)
        plt.show()

    else:
        plt.plot(t, w)
        plt.show()

    return w

def heun(f, a, b, alpha, n):
    h = (b-a) / n

    t = np.linspace(a, b, n+1)
    w = np.zeros(n+1)
    w[0] = alpha

    for i in range(n):
        w[i+1] = w[i] + (h/4) * (f(t[i], w[i]) + 3*f(t[i] + 2*h/3,
                                                     w[i] + (2*h/3) * f(t[i] + h/3, w[i] + (h/3) * f(t[i], w[i]))))

    plt.grid()
    if h > 0.1:
        plt.scatter(t, w, c='r')
        plt.plot(t, w)
        plt.show()

    else:
        plt.plot(t, w)
        plt.show()

    return w

def midpoint(f, a, b, alpha, n):
    h = (b-a) / n

    t = np.linspace(a, b, n+1)
    w = np.zeros(n+1)
    w[0] = alpha

    for i in range(n):
        w[i+1] = w[i] + h*f(t[i] + h/2, w[i] + (h/2)*f(t[i], w[i]))

    plt.grid()
    if h > 0.1:
        plt.scatter(t, w, c='r')
        plt.plot(t, w)
        plt.show()

    else:
        plt.plot(t, w)
        plt.show()

    return w

def rk4(f, a, b, alpha, n):
    h = (b-a) / n

    t = np.linspace(a, b, n+1)
    w = np.zeros(n+1)
    w[0] = alpha

    for i in range(n):
        k1 = h*f(t[i], w[i])
        k2 = h*f(t[i] + h/2, w[i] + k1/2)
        k3 = h*f(t[i] + h/2, w[i] + k2/2)
        k4 = h*f(t[i+1], w[i] + k3)

        w[i+1] = w[i] + (1/6) * (k1 + 2*k2 + 2*k3 + k4)

    plt.grid()
    if h > 0.1:
        plt.scatter(t, w, c='r')
        plt.plot(t, w)
        plt.show()

    else:
        plt.plot(t, w)
        plt.show()

    return w