import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)


def backward(f, alpha, g1, g2, xa, xb, ta, tb, h, k, plot=None):
    N = round((xb-xa) / h)
    M = round((tb-ta) / k)

    l = alpha**2 * k / h**2
    x = np.linspace(xa, xb, N+1)
    t = np.linspace(ta, tb, M+1)
    w = np.zeros((len(t), len(x)))

    w[0] = f(x)
    w[1:, 0] = g1(t[1:])
    w[1:, -1] = g2(t[1:])

    # declare tri-diagonal matrix
    J = diags([1 + 2*l, -l, -l], [0, -1, 1], shape=(N-1, N-1)).tocsc()

    # vector to allocate all the right hand vectors of the system that needs to be solved for each i
    v = np.zeros((M,N-1))

    # iterate over v to update the values in a loop
    vb = np.zeros(N-1)
    for j in range(1, M+1):
        vb.fill(0)
        vb[0] = w[j,0]
        vb[-1] = w[j,N]
        v[j-1] = w[j-1,1:N] + l*vb
        w[j,1:-1] = spsolve(J, v[j-1])

    if plot is not None:
        X,T = np.meshgrid(x,t)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, T, w, cmap='plasma')  # choose a colormap you like
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_zlabel('u(x, t)')
        plt.show()

    return w

