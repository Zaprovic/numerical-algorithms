import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)


def csnatural(x, y):
    n = len(x) - 1  # so we can have a set of points from x0,x1,x2,...,xn

    # conditions for natural spline
    c0 = 0
    cn = 0

    h = np.zeros(n)
    B = np.zeros(n - 1)

    for i in range(n):
        h[i] = x[i + 1] - x[i]

    for i in range(n - 1):
        B[i] = 3 * (y[i + 2] - y[i + 1]) / h[i + 1] - 3 * (y[i + 1] - y[i]) / h[i]

    D = np.diag([2 * (h[i] + h[i + 1]) for i in range(n - 1)])
    U = np.diag([h[i] for i in range(1, n - 1)], k=1)
    L = np.diag([h[i] for i in range(1, n - 1)], k=-1)

    A = D + L + U

    # constants
    a = np.array(y)
    c = np.concatenate(([c0], np.linalg.solve(A, B), [cn]))
    b = np.array(
        [(a[i + 1] - a[i]) / h[i] - h[i] * (2 * c[i] + c[i + 1]) / 3 for i in range(n)]
    )
    d = np.array([(c[i + 1] - c[i]) / (3 * h[i]) for i in range(n)])

    S = np.column_stack((a[0:-1], b, c[0:-1], d))

    return S


def csclamped(x, y, f1, f2):
    """
    Parameters
    ----------
    x
    y
    f1
    f2

    Returns
    -------

    """
    n = len(x) - 1  # so we can have a set of points from x0,x1,x2,...,xn

    h = np.zeros(n)
    B = np.zeros(n - 1)  # array of the bj

    for i in range(n):
        h[i] = x[i + 1] - x[i]

    for i in range(1, n - 1):
        B[i] = 3 * (y[i + 2] - y[i + 1]) / h[i + 1] - 3 * (y[i + 1] - y[i]) / h[i]

    B[0] = (
        3 * (y[2] - y[1]) / h[1]
        - (3 / h[0]) * (y[1] - y[0])
        - (3 / (2 * h[0])) * (y[1] - y[0])
        + 3 * f1 / 2
    )

    B[-1] = (
        3 * (y[n] - y[n - 1]) / h[n - 1]
        - 3 * (y[n - 1] - y[n - 2]) / h[n - 2]
        + 3 * (y[n] - y[n - 1]) / (2 * h[n - 1])
        - 3 * f2 / 2
    )

    D = np.diag([2 * (h[i] + h[i + 1]) for i in range(n - 1)])
    D[0, 0] = 2 * (h[0] + h[1]) - h[0] / 2
    D[-1, -1] = 2 * (h[n - 2] + h[n - 1]) - h[n - 1] / 2

    U = np.diag([h[i] for i in range(1, n - 1)], k=1)
    # U[0, 1] = h[1]

    L = np.diag([h[i] for i in range(1, n - 1)], k=-1)
    # L[-1, 1] = h[n-2]

    A = D + L + U

    # constants
    a = np.array(y)
    c = np.linalg.solve(A, B)
    c0 = 3 * (a[1] - a[0]) / (2 * h[0] ** 2) - 3 * f1 / (2 * h[0]) - c[0] / 2
    cn = (
        -3 * (a[n] - a[n - 1]) / (2 * h[n - 1] ** 2)
        + 3 * f2 / (2 * h[n - 1])
        - c[n - 2] / 2
    )

    c = np.concatenate(([c0], c, [cn]))
    b = np.array(
        [(a[i + 1] - a[i]) / h[i] - h[i] * (2 * c[i] + c[i + 1]) / 3 for i in range(n)]
    )
    b[0] = f1
    d = np.array([(c[i + 1] - c[i]) / (3 * h[i]) for i in range(n)])
    S = np.column_stack((a[0:-1], b, c[0:-1], d))

    return S


def csextrapolated(x, y):
    n = len(x) - 1  # so we can have a set of points from x0,x1,x2,...,xn

    h = np.zeros(n)
    B = np.zeros(n - 1)

    for i in range(n):
        h[i] = x[i + 1] - x[i]

    for i in range(n - 1):
        B[i] = 3 * (y[i + 2] - y[i + 1]) / h[i + 1] - 3 * (y[i + 1] - y[i]) / h[i]

    D = np.diag([2 * (h[i] + h[i + 1]) for i in range(n - 1)])
    D[0, 0] = h[0] + h[0] ** 2 / h[1] + 2 * (h[0] + h[1])
    D[-1, -1] = 2 * h[n - 2] + 3 * h[n - 1] + h[n - 1] ** 2 / h[n - 2]

    U = np.diag([h[i] for i in range(1, n - 1)], k=1)
    U[0, 1] = -h[0] ** 2 / h[1] + h[1]

    L = np.diag([h[i] for i in range(1, n - 1)], k=-1)
    L[-1, 1] = h[n - 2] - h[n - 1] ** 2 / h[n - 2]

    A = D + L + U

    # constants
    a = np.array(y)
    c = np.linalg.solve(A, B)

    # conditions for extrapolated spline
    c0 = c[0] - (h[0] / h[1]) * (c[1] - c[0])
    cn = c[n - 2] + (h[n - 1] / h[n - 2]) * (c[n - 2] - c[n - 3])

    c = np.concatenate(([c0], c, [cn]))
    b = np.array(
        [(a[i + 1] - a[i]) / h[i] - h[i] * (2 * c[i] + c[i + 1]) / 3 for i in range(n)]
    )
    d = np.array([(c[i + 1] - c[i]) / (3 * h[i]) for i in range(n)])

    S = np.column_stack((a[0:-1], b, c[0:-1], d))

    return S


def csknown(x, y, f1, f2):
    n = len(x) - 1  # so we can have a set of points from x0,x1,x2,...,xn

    h = np.zeros(n)
    B = np.zeros(n - 1)

    for i in range(n):
        h[i] = x[i + 1] - x[i]

    for i in range(1, n - 1):
        B[i] = 3 * (y[i + 2] - y[i + 1]) / h[i + 1] - 3 * (y[i + 1] - y[i]) / h[i]

    B[0] = 3 * (y[2] - y[1]) / h[1] - 3 * (y[1] - y[0]) / h[0] - f1 * h[0] / 2
    B[-1] = (
        3 * (y[n] - y[n - 1]) / h[n - 1]
        - 3 * (y[n - 1] - y[n - 2]) / h[n - 2]
        - f2 * h[n - 1] / 2
    )

    D = np.diag([2 * (h[i] + h[i + 1]) for i in range(n - 1)])
    U = np.diag([h[i] for i in range(1, n - 1)], k=1)
    L = np.diag([h[i] for i in range(1, n - 1)], k=-1)

    A = D + L + U

    # conditions for known curvature spline
    c0 = 0.5 * f1
    cn = 0.5 * f2

    # constants
    a = np.array(y)
    c = np.concatenate(([c0], np.linalg.solve(A, B), [cn]))
    b = np.array(
        [(a[i + 1] - a[i]) / h[i] - h[i] * (2 * c[i] + c[i + 1]) / 3 for i in range(n)]
    )
    d = np.array([(c[i + 1] - c[i]) / (3 * h[i]) for i in range(n)])

    S = np.column_stack((a[0:-1], b, c[0:-1], d))

    return S


def csparabolic(x, y):
    n = len(x) - 1  # so we can have a set of points from x0,x1,x2,...,xn

    h = np.zeros(n)
    B = np.zeros(n - 1)

    for i in range(n):
        h[i] = x[i + 1] - x[i]

    for i in range(n - 1):
        B[i] = 3 * (y[i + 2] - y[i + 1]) / h[i + 1] - 3 * (y[i + 1] - y[i]) / h[i]

    D = np.diag([2 * (h[i] + h[i + 1]) for i in range(n - 1)])
    D[0, 0] = 3 * h[0] + 2 * h[1]
    D[-1, -1] = 3 * h[n - 1] + 2 * h[n - 2]

    U = np.diag([h[i] for i in range(1, n - 1)], k=1)
    L = np.diag([h[i] for i in range(1, n - 1)], k=-1)

    A = D + L + U

    # constants
    a = np.array(y)
    c = np.linalg.solve(A, B)
    c = np.concatenate(([c[0]], c, [c[-1]]))
    b = np.array(
        [(a[i + 1] - a[i]) / h[i] - h[i] * (2 * c[i] + c[i + 1]) / 3 for i in range(n)]
    )
    d = np.array([(c[i + 1] - c[i]) / (3 * h[i]) for i in range(n)])

    S = np.column_stack((a[0:-1], b, c[0:-1], d))

    return S


def csplot(x, y, S, title=None, size=None):
    S = np.fliplr(S)
    n = len(x) - 1
    plt.style.use("ggplot")

    if size is not None:
        plt.figure(figsize=size)

    for i in range(n):
        xx = np.linspace(x[i], x[i + 1], 1500)
        yy = np.polyval(S[i, :], xx - x[i])

        plt.plot(xx, yy, linewidth=2.5)

    plt.scatter(x, y, c="k")

    if title is not None:
        plt.title(title)

    else:
        plt.title("Cubic spline interpolation")

    plt.show()
