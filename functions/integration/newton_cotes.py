import numpy as np


def closed(f, a, b, n):
    h = (b - a) / n

    # list comprehension
    x = np.array([a + i * h for i in range(n + 1)])
    y = f(x)

    A = np.zeros((n + 1, n + 1))

    for i in range(n + 1):
        A[i, :] = np.power(x, i)

    # list comprehension
    # b = np.array([(np.power(b, i+1)-np.power(a, i+1))/(i+1)
    #              for i in range(n+1)])

    b = np.array([np.power(b, i) / i - np.power(a, i) / i for i in range(1, n + 2)])

    # pesos de cuadratura
    r = np.linalg.solve(A, b)
    S = np.dot(r, y)

    return S


def open(f, a, b, n):
    h = (b - a) / (n + 2)

    x = np.array([a + i * h for i in range(n + 3)])
    y = f(x)

    A = np.zeros((n + 1, n + 1))

    for i in range(n + 1):
        A[i, :] = np.power(x[1:-1], i)

    b = np.array([np.power(b, i) / i - np.power(a, i) / i for i in range(1, n + 2)])
    r = np.linalg.solve(A, b)
    S = np.dot(r, y[1:-1])

    return S
