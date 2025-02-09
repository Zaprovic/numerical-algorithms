import numpy as np


def gseidel(A, b, x0, epsilon, max):
    A = np.array(A)
    b = np.array(b).reshape(-1, 1)
    x0 = np.array(x0).reshape(-1, 1)

    n = len(A)
    L = np.tril(A)
    U = A - L

    Tgs = -np.linalg.inv(L) @ U
    cgs = np.linalg.inv(L) @ b

    for i in range(max):
        x = Tgs @ x0 + cgs

        if np.allclose(x0, x, rtol=epsilon):
            return x

        x0 = x

    eig = np.linalg.eigvals(Tgs)
    eig = list(set([np.abs(i) for i in eig]))
    rho = np.max(eig)

    if rho > 1:
        print(f"The system will not converge\nSpectral radius is {rho}")
    else:
        print(f"The system will converge\nSpectral radius is {rho}\n")

    return x
