import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-dark')


def finite_differences(p, q, r, a, b, alpha, beta, n, plot=None):
    """
    Parameters
    Equation of the form y" = p(x)y' + q(x)y + r(x), y(a) = alpha, y(b) = beta
    ----------
    p
    q
    r
    a
    b
    alpha
    beta
    n: Number of points
    plot: Can be 'scatter' or 'continuous'

    Returns
    -------
    """
    h = (b - a) / n

    x = np.linspace(a, b, n+1)
    U = np.zeros((n - 1, n - 1))

    D = np.diag([-(2 / h ** 2 + q(x[i])) for i in range(1, n)])
    L = np.diag([1 / h ** 2 + (1 / (2 * h))
                 * p(x[i]) for i in range(2, n)], k=-1)
    U = np.diag([1 / h ** 2 - (1 / (2 * h))
                 * p(x[i]) for i in range(1, n-1)], k=1)

    # # Tridiagonal matrix
    M = D + L + U

    # vector b of the form Mx = b
    b = np.array([r(x[i]) for i in range(1, n)])
    b[0] = r(x[1]) - alpha * (1 / h**2 + p(x[1]) / (2*h))
    b[-1] = r(x[n-1]) - beta * (1 / h**2 - p(x[n-1]) / (2*h))

    # solution
    y = np.linalg.solve(M, b)
    y = np.concatenate([[alpha], y, [beta]])
    points = [(i, j) for i, j in list(zip(x, y))]

    if plot is not None:
        x = [p[0] for p in points]
        y = [p[1] for p in points]

        plt.figure(figsize=(10, 10))
        plt.grid()

        if plot == 'scatter':
            plt.scatter(x, y, c='k')
            plt.show()

        if plot == 'continuous':
            plt.plot(x, y, marker='.', c='k')
            plt.show()

    return x, y
