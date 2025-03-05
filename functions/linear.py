import numpy as np
from typing import Tuple, Optional, Dict, Any


def jacobi(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    epsilon: float = 1e-6,
    max_iter: int = 100,
) -> Tuple[np.ndarray, int, float]:
    """
    Solve a system of linear equations using the Jacobi iterative method.

    The Jacobi method solves the system Ax = b by iteratively updating the solution
    vector using the formula: x^(k+1) = D^(-1)(b - (L+U)x^(k)), where D is the diagonal
    of A, and L and U are the strictly lower and upper triangular parts of A.

    Parameters
    ----------
    A : np.ndarray
        The coefficient matrix (must be square with non-zero diagonal elements)
    b : np.ndarray
        The right-hand side vector
    x0 : np.ndarray
        Initial guess for the solution vector
    epsilon : float, optional
        Convergence tolerance (default: 1e-6)
    max_iter : int, optional
        Maximum number of iterations (default: 100)

    Returns
    -------
    Tuple[np.ndarray, int, float]
        A tuple containing:
        - The solution vector
        - Number of iterations performed
        - Final relative error

    Raises
    ------
    ValueError
        If A is not square, dimensions don't match, or A has zero diagonal elements
    """

    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1, 1)
    x0 = np.array(x0, dtype=float).reshape(-1, 1)

    n = A.shape[0]
    if A.shape[1] != n:
        raise ValueError("Matrix A must be square")
    if b.shape[0] != n:
        raise ValueError(
            f"Dimensions mismatch: A is {n}x{n}, but b has {b.shape[0]} rows"
        )
    if x0.shape[0] != n:
        raise ValueError(
            f"Dimensions mismatch: A is {n}x{n}, but x0 has {x0.shape[0]} elements"
        )

    D = np.diag(np.diag(A))
    U = np.triu(A) - D
    L = np.tril(A) - D

    if np.any(np.diag(A) == 0):
        raise ValueError("Jacobi method requires non-zero diagonal elements")

    Tj = -np.linalg.inv(D) @ (L + U)
    cj = np.linalg.inv(D) @ b

    spectral_radius = max(abs(np.linalg.eigvals(Tj)))
    will_converge = spectral_radius < 1

    x = x0.copy()
    iterations = 0
    error = float("inf")

    for i in range(max_iter):
        x_new = Tj @ x + cj
        error = np.linalg.norm(x_new - x) / np.linalg.norm(x_new)
        x = x_new
        iterations = i + 1

        if error < epsilon:
            break

    return x, iterations, error


def gseid(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    epsilon: float = 1e-6,
    max_iter: int = 100,
) -> Tuple[np.ndarray, int, float]:
    """
    Solve a system of linear equations using the Gauss-Seidel iterative method.

    The Gauss-Seidel method solves the system Ax = b by iteratively updating the solution
    vector using the formula: x^(k+1) = (D+L)^(-1)(b - Ux^(k)), where D is the diagonal
    of A, L is the strictly lower triangular part, and U is the strictly upper triangular part of A.

    Parameters
    ----------
    A : np.ndarray
        The coefficient matrix (must be square with non-zero diagonal elements)
    b : np.ndarray
        The right-hand side vector
    x0 : np.ndarray
        Initial guess for the solution vector
    epsilon : float, optional
        Convergence tolerance (default: 1e-6)
    max_iter : int, optional
        Maximum number of iterations (default: 100)

    Returns
    -------
    Tuple[np.ndarray, int, float]
        A tuple containing:
        - The solution vector
        - Number of iterations performed
        - Final relative error

    Raises
    ------
    ValueError
        If A is not square, dimensions don't match, or A has zero diagonal elements
    """

    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1, 1)
    x0 = np.array(x0, dtype=float).reshape(-1, 1)

    n = A.shape[0]
    if A.shape[1] != n:
        raise ValueError("Matrix A must be square")
    if b.shape[0] != n:
        raise ValueError(
            f"Dimensions mismatch: A is {n}x{n}, but b has {b.shape[0]} rows"
        )
    if x0.shape[0] != n:
        raise ValueError(
            f"Dimensions mismatch: A is {n}x{n}, but x0 has {x0.shape[0]} elements"
        )

    D = np.diag(np.diag(A))
    L = np.tril(A) - D
    U = np.triu(A) - D

    if np.any(np.diag(A) == 0):
        raise ValueError("Gauss-Seidel method requires non-zero diagonal elements")

    Tgs = -np.linalg.inv(D + L) @ U
    cgs = np.linalg.inv(D + L) @ b

    spectral_radius = max(abs(np.linalg.eigvals(Tgs)))
    will_converge = spectral_radius < 1

    x = x0.copy()
    iterations = 0
    error = float("inf")

    for i in range(max_iter):
        x_new = Tgs @ x + cgs
        error = np.linalg.norm(x_new - x) / np.linalg.norm(x_new)
        x = x_new
        iterations = i + 1

        if error < epsilon:
            break

    return x, iterations, error


def sor(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    omega: float,
    epsilon: float = 1e-6,
    max_iter: int = 100,
) -> Tuple[np.ndarray, int, float]:
    """
    Solve a system of linear equations using the Successive Over-Relaxation (SOR) iterative method.

    The SOR method solves the system Ax = b by iteratively updating the solution
    vector using the formula: x^(k+1) = (1-ω)x^(k) + ω(D+ωL)^(-1)(b - Ux^(k) - (1-ω)Dx^(k)),
    where D is the diagonal of A, L is the strictly lower triangular part,
    U is the strictly upper triangular part of A, and ω is the relaxation parameter.

    Parameters
    ----------
    A : np.ndarray
        The coefficient matrix (must be square with non-zero diagonal elements)
    b : np.ndarray
        The right-hand side vector
    x0 : np.ndarray
        Initial guess for the solution vector
    omega : float
        Relaxation parameter (0 < omega < 2)
    epsilon : float, optional
        Convergence tolerance (default: 1e-6)
    max_iter : int, optional
        Maximum number of iterations (default: 100)

    Returns
    -------
    Tuple[np.ndarray, int, float]
        A tuple containing:
        - The solution vector
        - Number of iterations performed
        - Final relative error

    Raises
    ------
    ValueError
        If A is not square, dimensions don't match, A has zero diagonal elements,
        or omega is not in the range (0, 2)
    """

    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1, 1)
    x0 = np.array(x0, dtype=float).reshape(-1, 1)

    # Check if omega is in valid range
    if omega <= 0 or omega >= 2:
        raise ValueError("Relaxation parameter omega must be in the range (0, 2)")

    n = A.shape[0]
    if A.shape[1] != n:
        raise ValueError("Matrix A must be square")
    if b.shape[0] != n:
        raise ValueError(
            f"Dimensions mismatch: A is {n}x{n}, but b has {b.shape[0]} rows"
        )
    if x0.shape[0] != n:
        raise ValueError(
            f"Dimensions mismatch: A is {n}x{n}, but x0 has {x0.shape[0]} elements"
        )

    D = np.diag(np.diag(A))
    L = np.tril(A) - D
    U = np.triu(A) - D

    if np.any(np.diag(A) == 0):
        raise ValueError("SOR method requires non-zero diagonal elements")

    # SOR iteration matrix
    Tsor = np.linalg.inv(D + omega * L) @ ((1 - omega) * D - omega * U)
    csor = omega * np.linalg.inv(D + omega * L) @ b

    spectral_radius = max(abs(np.linalg.eigvals(Tsor)))
    will_converge = spectral_radius < 1

    x = x0.copy()
    iterations = 0
    error = float("inf")

    for i in range(max_iter):
        x_new = Tsor @ x + csor
        error = np.linalg.norm(x_new - x) / np.linalg.norm(x_new)
        x = x_new
        iterations = i + 1

        if error < epsilon:
            break

    return x, iterations, error
