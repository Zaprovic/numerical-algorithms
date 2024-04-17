import numpy as np


def euler(F, a, b, alpha, n):
    h = (b-a)/n

    t = np.linspace(a, b, n+1)
    W = np.zeros((n+1, len(alpha)))
    W[0] = alpha

    for i in range(n):
        W[i+1] = W[i] + h*np.array(F(t[i], W[i]))

    return W

def eulermod(F, a, b, alpha, n):
    h = (b-a)/n

    t = np.linspace(a, b, n+1)
    W = np.zeros((n+1, len(alpha)))
    W[0] = alpha

    for i in range(n):
        W[i+1] = W[i] + (h/2) * (np.array(F(t[i], W[i])) +
                                 np.array(F(t[i+1], W[i] + h*np.array(F(t[i], W[i])))))

    return W

def rk4(F, a, b, alpha, n):
    h = (b-a)/n

    t = np.linspace(a, b, n+1)
    W = np.zeros((n+1, len(alpha)))
    W[0] = alpha

    for i in range(n):
        k1 = h*np.array(F(t[i], W[i]))
        k2 = h*np.array(F(t[i] + h/2, W[i] + k1/2))
        k3 = h*np.array(F(t[i] + h/2, W[i] + k2/2))
        k4 = h*np.array(F(t[i+1], W[i] + k3))

        W[i+1] = W[i] + (1/6) * (k1 + 2*k2 + 2*k3 + k4)

    return W