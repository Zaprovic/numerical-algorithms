import numpy as np

np.set_printoptions(suppress=True)


def backward(f, g1, g2, xa, xb, ta, tb, h, k):
    N = round((xb-xa) / h)
    M = round((tb-ta) / k)

    lmbda = alpha**2 * k / h**2
    x = np.linspace(xa, xb, N+1)
    t = np.linspace(ta, tb, M+1)
    w = np.zeros((len(t), len(x)))

    w[0] = [f(x[i]) for i in range(N+1)]
    w[1:, 0] = [g1(t[i]) for i in range(1, M+1)]
    w[1:, -1] = [g2(t[i]) for i in range(1, M+1)]

    # declare tridiagonal matrix
    D = np.diag([1 + 2*lmbda for i in range(N-1)])
    L = np.diag([-lmbda for i in range(N-2)], k=-1)
    U = np.diag([-lmbda for i in range(N-2)], k=1)
    T = D + L + U

    v = [[w[j-1, i] + lmbda*w[j, i-1]
          for i in range(1, N)] for j in range(1, M+1)]

    for i in range(len(v)):
        v[i][-1] = v[i][-1] + lmbda*w[i+1, -1]

    xs = [np.linalg.solve(T, k) for k in v]

    for i in range(1, N-1):
        w[i, 1:-1] = xs[i-1]

    return w


# x interval
xa = 0
xb = 2

# t interval
ta = 0
tb = 0.8

alpha = 4
h = 0.5
k = 0.2

# boundary conditions


def f(x): return np.sin(np.pi*x/4)
def g1(t): return 0
def g2(t): return np.exp(-np.pi**2 * t)


S = backward(f, g1, g2, xa, xb, ta, tb, h, k)
print(S)
