import numpy as np


def csparabolic(x, y):
    n = len(x) - 1  # so we can have a set of points from x0,x1,x2,...,xn

    h = np.zeros(n)
    B = np.zeros(n-1)

    for i in range(n):
        h[i] = x[i+1] - x[i]

    for i in range(n-1):
        B[i] = 3*(y[i+2] - y[i+1])/h[i+1] - 3*(y[i+1] - y[i])/h[i]

    D = np.diag([2*(h[i] + h[i + 1]) for i in range(n - 1)])
    D[0, 0] = 3*h[0] + 2*h[1]
    D[-1, -1] = 3*h[n-1] + 2*h[n-2]

    U = np.diag([h[i] for i in range(1, n - 1)], k=1)
    L = np.diag([h[i] for i in range(1, n - 1)], k=-1)

    A = D + L + U

    # constants
    a = np.array(y)
    c = np.linalg.solve(A, B)
    c = np.concatenate(([c[0]], c, [c[-1]]))
    b = np.array([(a[i+1] - a[i])/h[i] - h[i] *
                 (2*c[i] + c[i+1])/3 for i in range(n)])
    d = np.array([(c[i+1] - c[i])/(3*h[i]) for i in range(n)])

    S = np.column_stack((a[0:-1], b, c[0:-1], d))

    return S
