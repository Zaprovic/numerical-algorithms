import numpy as np


def newton_cotes(f, a, b, n):
    h = (b-a)/n

    x = [a + i*(b-a)/n for i in range(n+1)]

    A = np.zeros((n+1, n+1))

    for i in range(n+1):
        A[i, :] = np.power(x, i)

    b = [(np.power(b, i+1)-np.power(a, i+1))/(i+1) for i in range(n+1)]

    r = np.linalg.solve(A, b)
    S = np.dot(r, f(x))

    return S
