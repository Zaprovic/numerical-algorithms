import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(suppress=True)


def forward(f, alpha, g1, g2, xa, xb, ta, tb, h, k, plot=None):
    N = round((xb-xa) / h)
    M = round((tb-ta) / k)

    l = alpha**2 * k / h**2
    x = np.linspace(xa, xb, N+1)
    t = np.linspace(ta, tb, M+1)
    w = np.zeros((len(t), len(x)))

    w[0] = f(x)
    w[1:, 0] = [g1(t[i]) for i in range(1, M+1)]
    w[1:, -1] = [g2(t[i]) for i in range(1, M+1)]

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
