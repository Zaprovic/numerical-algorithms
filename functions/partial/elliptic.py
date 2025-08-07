from typing import Callable, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix, diags, lil_matrix
from scipy.sparse.linalg import spsolve


def findiff(
    f: Callable[[float, float], float],
    g1: Callable[[float], Union[float, np.ndarray]],
    g2: Callable[[float], Union[float, np.ndarray]],
    h1: Callable[[float], Union[float, np.ndarray]],
    h2: Callable[[float], Union[float, np.ndarray]],
    xa: float,
    xb: float,
    ya: float,
    yb: float,
    h: float,
    k: float,
    plot=None,
):
    """
    Elliptic partial differential equation in the form
                uxx + uyy = f(x,y),     xa<=x<=xb, ya<=y<=yb
                                        u(xa,y) = g1(y), u(xb,y) = g2(y)
                                        u(x,ya) = h1(x), u(x,yb) = h2(x)

    Parameters
    ----------
    plot: If not None, then it will show 3D surface plot
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

    # Calculate number of interior points
    N = int((xb - xa) / h) - 1  # Interior points in x-direction
    M = int((yb - ya) / k) - 1  # Interior points in y-direction

    # Create grid points
    x = np.linspace(xa, xb, N + 2)  # Include boundary points
    y = np.linspace(ya, yb, M + 2)  # Include boundary points

    # Initialize solution matrix
    w = np.zeros((len(x), len(y)))

    # Apply boundary conditions
    w[0, :] = g1(y)  # Left boundary: u(xa, y)
    w[-1, :] = g2(y)  # Right boundary: u(xb, y)
    w[:, 0] = h1(x)  # Bottom boundary: u(x, ya)
    w[:, -1] = h2(x)  # Top boundary: u(x, yb)

    # Total number of interior points
    n_interior = N * M

    if n_interior == 0:
        print("No interior points to solve")
        return w

    # Create coefficient matrix A for the linear system Au = b
    A = lil_matrix((n_interior, n_interior))
    b = np.zeros(n_interior)

    # Coefficients for the finite difference stencil
    alpha = k**2 / h**2
    beta = h**2 / k**2
    gamma = -2 * (1 + alpha)

    # Fill the coefficient matrix using lexicographic ordering
    # Interior points are numbered: (i,j) -> i*M + j for i=1...N, j=1...M
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            row = (i - 1) * M + (j - 1)  # Current point index

            # Central point coefficient
            A[row, row] = gamma

            # Left neighbor (i-1, j)
            if i > 1:
                col = (i - 2) * M + (j - 1)
                A[row, col] = alpha
            else:
                # Boundary contribution
                b[row] -= alpha * w[0, j]

            # Right neighbor (i+1, j)
            if i < N:
                col = i * M + (j - 1)
                A[row, col] = alpha
            else:
                # Boundary contribution
                b[row] -= alpha * w[N + 1, j]

            # Bottom neighbor (i, j-1)
            if j > 1:
                col = (i - 1) * M + (j - 2)
                A[row, col] = beta
            else:
                # Boundary contribution
                b[row] -= beta * w[i, 0]

            # Top neighbor (i, j+1)
            if j < M:
                col = (i - 1) * M + j
                A[row, col] = beta
            else:
                # Boundary contribution
                b[row] -= beta * w[i, M + 1]

            # Right-hand side from source term
            b[row] += h**2 * f(x[i], y[j])

    # Convert to CSR format for efficient solving
    A = A.tocsr()

    # Solve the linear system
    u_interior = spsolve(A, b)

    # Place interior solution back into the grid
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            idx = (i - 1) * M + (j - 1)
            w[i, j] = u_interior[idx]

    # Plot if requested
    if plot is not None:
        X, Y = np.meshgrid(x, y)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(X, Y, w.T, cmap="plasma", alpha=0.8)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("u(x, y)")
        ax.set_title("Solution of 2D Poisson Equation")
        plt.colorbar(surf)
        plt.show()

    return w
