import numpy as np

def jacobi(A,b,x0,epsilon,max):
    A = np.array(A)
    b = np.array(b).reshape(-1, 1)
    x0 = np.array(x0).reshape(-1, 1)

    n = len(A)
    D = np.diag(np.diag(A))
    U = (np.triu(A) - D)
    L = (np.tril(A) - D)

    Tj = -np.linalg.inv(D) @ (L+U)
    cj = np.linalg.inv(D) @ b

    for i in range(max):
        x = Tj @ x0 + cj

        if np.allclose(x0,x, rtol=epsilon):
            return x

        x0 = x

    eig = np.linalg.eigvals(Tj)

    eig = list(set([np.abs(i) for i in eig]))
    rho = np.max(eig)

    if rho > 1:
        print(f'The system will not converge\nSpectral radius is {rho}')

    else:
        print(f'The system will converge\nSpectral radius is {rho}\n')

    return x