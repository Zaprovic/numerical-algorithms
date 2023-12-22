import numpy as np

np.set_printoptions(suppress=True)


def backward(f, alpha, g1, g2, xa, xb, ta, tb, h, k):
    N = round((xb-xa) / h)
    M = round((tb-ta) / k)

    l = alpha**2 * k / h**2
    x = np.linspace(xa, xb, N+1)
    t = np.linspace(ta, tb, M+1)
    w = np.zeros((len(t), len(x)))

    w[0] = [f(x[i]) for i in range(N+1)]
    w[1:, 0] = [g1(t[i]) for i in range(1, M+1)]
    w[1:, -1] = [g2(t[i]) for i in range(1, M+1)]

    # declare tri-diagonal matrix
    D = np.diag([1 + 2*l for i in range(N-1)])
    L = np.diag([-l for i in range(N-2)], k=-1)
    U = np.diag([-l for i in range(N-2)], k=1)
    T = D + L + U

    # vector to allocate all the right hand vectors of the system that needs to be solved for each i
    v = np.zeros((M,N-1))

    # iterate over v to update the values in a loop
    for j in range(1,M+1):
        vb = np.zeros(N-1)
        vb[0] = w[j,0]
        vb[-1] = w[j,N]

        v[j-1] = [w[j-1,i] for i in range(1,N)] + l*vb
        w[j,1:-1] = np.linalg.solve(T, v[j-1])

    return w

