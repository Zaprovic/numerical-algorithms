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



