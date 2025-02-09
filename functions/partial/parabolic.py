import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


def forward(f, alpha, g1, g2, xa, xb, ta, tb, h, k, plot=None):
    N = round((xb-xa) / h)
    M = round((tb-ta) / k)

    l = alpha**2 * k / h**2
    x = np.linspace(xa, xb, N+1)
    t = np.linspace(ta, tb, M+1)
    w = np.ones((len(t), len(x)))

    w[0] = f(x)
    w[1:, 0] = g1(t[1:])
    w[1:, -1] = g2(t[1:])

    for j in range(M):
        for i in range(1, N):
            w[j+1, i] = l*(w[j, i+1] - 2*w[j, i] + w[j, i-1]) + w[j, i]

    if plot is not None:
        X, T = np.meshgrid(x, t)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, T, w, cmap='plasma')  # choose a colormap you like
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_zlabel('u(x, t)')
        plt.show()

    return w


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

    # declare tridiagonal matrix
    J = diags([1 + 2*l, -l, -l], [0, -1, 1], shape=(N-1, N-1)).tocsc()

    # vector to allocate all the right hand vectors of the system that needs to be solved for each i
    v = np.zeros((M, N-1))

    # iterate over v to update the values in a loop
    vb = np.zeros(N-1)
    for j in range(1, M+1):
        vb.fill(0)
        vb[0] = w[j, 0]
        vb[-1] = w[j, N]
        v[j-1] = w[j-1, 1:N] + l*vb
        w[j, 1:-1] = spsolve(J, v[j-1])

    if plot is not None:
        X, T = np.meshgrid(x, t)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, T, w, cmap='plasma')
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_zlabel('u(x, t)')
        plt.show()

    return w


def crnich(f, alpha, g1, g2, xa, xb, ta, tb, h, k, plot=None):
    N = round((xb-xa) / h)
    M = round((tb-ta) / k)

    l = alpha**2 * k / h**2
    x = np.linspace(xa, xb, N+1)
    t = np.linspace(ta, tb, M+1)
    w = np.zeros((len(t), len(x)))

    w[0] = f(x)
    w[1:, 0] = g1(t[1:])
    w[1:, -1] = g2(t[1:])

    # optimized way to declare tri-diagonal matrix using scipy sparse module
    T = diags([1+l, -l/2, -l/2], [0, -1, 1], shape=(N-1, N-1)).tocsc()

    # vector to allocate all the right hand vectors of the system that needs to be solved for each i
    v = np.zeros((M, N-1))

    # iterate over v to update the values in a loop
    for j in range(M):
        vb = np.zeros(N-1)
        vb[0] = w[j+1, 0] + w[j, 0]
        vb[-1] = w[j+1, N] + w[j, N]

        J = diags([1-l, l/2, l/2], [0, -1, 1], shape=(N-1, N-1)).tocsc()
        v[j] = J.dot(w[j, 1:-1]) + (l/2)*vb
        w[j+1, 1:-1] = spsolve(T, v[j])

    if plot is not None:
        X, T = np.meshgrid(x, t)
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, T, w, cmap='plasma')
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_zlabel('u(x, t)')
        plt.show()

    return w
