import numpy as np


def csknown(x, y, f1, f2):
    n = len(x) - 1  # so we can have a set of points from x0,x1,x2,...,xn

    h = np.zeros(n)
    B = np.zeros(n-1)

    for i in range(n):
        h[i] = x[i+1] - x[i]

    for i in range(1, n-1):
        B[i] = 3*(y[i+2] - y[i+1])/h[i+1] - 3*(y[i+1] - y[i])/h[i]

    B[0] = 3*(y[2] - y[1])/h[1] - 3*(y[1] - y[0])/h[0] - f1*h[0]/2
    B[-1] = 3*(y[n] - y[n-1])/h[n-1] - 3*(y[n-1] - y[n-2])/h[n-2] - f2*h[n-1]/2

    D = np.diag([2*(h[i] + h[i + 1]) for i in range(n - 1)])
    U = np.diag([h[i] for i in range(1, n - 1)], k=1)
    L = np.diag([h[i] for i in range(1, n - 1)], k=-1)

    A = D + L + U

    # conditions for known curvature spline
    c0 = 0.5*f1
    cn = 0.5*f2

    # constants
    a = np.array(y)
    c = np.concatenate(([c0], np.linalg.solve(A, B), [cn]))
    b = np.array([(a[i+1] - a[i])/h[i] - h[i] *
                 (2*c[i] + c[i+1])/3 for i in range(n)])
    d = np.array([(c[i+1] - c[i])/(3*h[i]) for i in range(n)])

    S = np.column_stack((a[0:-1], b, c[0:-1], d))

    return S
