import numpy as np
from scipy.sparse import diags, bmat, csr_matrix, lil_matrix, block_diag
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve


def findiff(f, g1, g2, h1, h2, xa, xb, ya, yb, h, k ,plot=None):
    """
    Elliptic partial differential equation in the form
                uxx + uyy = f(x,y),     xa<=x<=xb, ya<=y<=yb
                                        u(xa,y) = g1(y), u(xb,y) = g2(y)
                                        u(x,ya) = h1(x), u(x,yb) = h2(x)

    Parameters
    ----------
    plot: If not None, then it will shoe 3D surface plot
    f: function f(x,y)
    g1: g1(y) --> boundary condition for u(xa, y)
    g2: g2(y) --> boundary condition for u(xb, y)
    h1: h1(x) --> boundary condition for u(x, ya)
    h2: h2(x) --> boundary condition for u(x, yb)
    xa: interval of [xa, xb]
    xb: interval of [xa, xb]
    ya: interval of [ya, yb]
    yb: interval of [ya, yb]
    h: step size for the variable x
    k: step size for the variable y

    Returns --> w: matrix solution
    -------
    """

    N = round((xb-xa)/h)
    M = round((yb-ya)/k)

    x = np.linspace(xa, xb, N+1)
    y = np.linspace(ya, yb, M+1)
    w = lil_matrix((len(x),len(y)), dtype=float)

    w[0] = g1(y)
    w[-1] = g2(y)

    w[1:-1,0] = h1(x[1:-1])
    w[1:-1,-1] = h2(x[1:-1])

    l = np.power(h/k,2)

    trd = diags([-2*(1+l),1,1], [0,-1,1], shape=(N-1,N-1), format="csr")
    off = diags([l,0,0], [0,-1,1], shape=(N-1,N-1), format="csr")

    T = csr_matrix(bmat([
        [None]*(M-1-M) + [trd, off] + [None]*(M-1-2),
        *[[None]*(i-3) + [off, trd, off] + [None]*(M-1-i) for i in range(3, M)],
        [None] * (M-1-2) + [off, trd] + [None]*(M-1-M)
    ]))

    # vector to store the solutions for f(xi,yj)
    F = csr_matrix([f(x[i],y[j]) for j in range(1,M) for i in range(1,N)]).toarray().ravel()

    G = csr_matrix([
        [w[0, 1] + l * w[1, 0]] + [0 + l * w[i, 0] for i in range(2, N - 1)] + [w[N, 1] + l * w[N - 1, 0]],
         *[[w[0, j]] + [0 for _ in range(2, N - 1)] + [w[N, 2]] for j in range(2, M - 1)],
         [w[0, 3] + l * w[1, M]] + [0 + l * w[i, M] for i in range(2, N - 1)] + [w[N, M - 1] + l * w[N - 1, M]
        ]]).toarray().reshape(((N-1)*(M-1)))

    S = np.power(h,2)*F - G

    xs = lil_matrix(spsolve(T,S)).reshape((M-1,N-1)).transpose()
    w[1:-1,1:-1] = xs

    if plot is not None:
        X, Y = np.meshgrid(x, y)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, w.toarray().T, cmap='plasma')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('u(x, y)')
        plt.show()

    return T.shape