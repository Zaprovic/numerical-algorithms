import matplotlib.pyplot as plt
import numpy as np


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
