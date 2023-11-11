import numpy as np


def euler_vector(F, a, b, alpha, n):
    h = (b-a)/n

    t = np.linspace(a, b, n+1)
    W = np.zeros((n+1, len(alpha)))
    W[0] = alpha

    for i in range(n):
        W[i+1] = W[i] + (h/2) * (np.array(F(t[i], W[i])) +
                                 np.array(F(t[i+1], W[i] + h*np.array(F(t[i], W[i])))))

    return W






