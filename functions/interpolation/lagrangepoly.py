import numpy as np

np.set_printoptions(suppress=True)


def lagrange(x, y):
    n = len(x)
    y = np.array(y)

    # Calculating denominators
    d = np.prod([[x[i] - x[j] for j in range(n) if j != i] for i in range(n)], axis=1)

    # # denominators
    # d = []
    # for i in range(n):
    #     c = 1
    #     for j in range(n):
    #         if i != j:
    #             c *= (x[i] - x[j])
    #     d.append(c)

    # d = np.array(d)

    # numerators
    T = np.array([[v for idx, v in enumerate(x) if idx != i] for i in range(n)])

    Tk = np.array([[[1, -i] for i in T[k]] for k in range(n)])

    Lk = []

    for idx, t in enumerate(Tk):
        r = [1]
        for seq in t:
            r = np.convolve(r, seq)
        Lk.append(r / d[idx])

    # Lagrange coefficients
    Lk = np.array(Lk)

    c = 0
    for i in range(n):
        c += Lk[i] * y[i]

    return c, Lk
