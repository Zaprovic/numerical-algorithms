import numpy as np

np.set_printoptions(suppress=True)


def forward2(f, g1, g2, xa, xb, ta, tb, h, k):
    N = round((xb-xa) / h)
    M = round((tb-ta) / k)

    lmbda = alpha**2 * k / h**2
    x = np.linspace(xa, xb, N+1)
    t = np.linspace(ta, tb, M+1)
    w = np.zeros((len(t), len(x)))

    w[0] = [f(x[i]) for i in range(N+1)]
    w[1:, 0] = [g1(t[i]) for i in range(1, M+1)]
    w[1:, -1] = [g2(t[i]) for i in range(1, M+1)]

    # w[1, 1] = lmbda * w[0, 2] + (1 - 2*lmbda)*w[0, 1] + lmbda*w[0, 0]
    # w[1, 2] = lmbda * w[0, 3] + (1 - 2*lmbda)*w[0, 2] + lmbda*w[0, 1]
    # w[1, 3] = lmbda * w[0, 4] + (1 - 2*lmbda)*w[0, 3] + lmbda*w[0, 2]

    for j in range(M):
        for i in range(1, N):
            w[j+1, i] = lmbda * w[j, i+1] + \
                (1-2*lmbda)*w[j, i] + lmbda*w[j, i-1]

    return w


# x interval
xa = 0
xb = 2

# t interval
ta = 0
tb = 0.6

alpha = 4
h = 0.5
k = 0.2

# boundary conditions


def f(x): return np.sin(np.pi*x/4)
def g1(t): return 0
def g2(t): return np.exp(-np.pi**2 * t)


S = forward2(f, g1, g2, xa, xb, ta, tb, h, k)
print(S)
