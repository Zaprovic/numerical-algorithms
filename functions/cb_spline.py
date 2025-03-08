import numpy as np
import numpy.typing as npt
from typing import Literal, Optional, Union, Tuple
from scipy.sparse import diags

np.set_printoptions(precision=4)


class CubicSpline:
    """
    CubicSpline: A class for cubic spline interpolation.

    This class computes cubic spline interpolation of a set of data points and evaluates
    the resulting spline at query points. Various boundary conditions are supported.

    Parameters
    ----------
    x : array_like
        1-D array of independent variable values in strictly ascending order.
    y : array_like
        1-D array of dependent variable values, same length as x.
    bc_type : {'natural', 'clamped', 'known', 'parabolic', 'extrapolated'}, optional
        Specifies the boundary condition type. Options:
        - 'natural': Second derivatives at endpoints are zero (default)
        - 'clamped': First derivatives at endpoints are specified by bc
        - 'known': Second derivatives at endpoints are specified by bc
        - 'parabolic': Boundary segments are parabolic (not cubic)
        - 'extrapolated': Extrapolates the first derivatives at endpoints
    bc : tuple(float, float), optional
        Tuple containing boundary condition values.
        For 'clamped': values of the first derivatives at the endpoints.
        For 'known': values of the second derivatives at the endpoints.
        Default is (0.0, 0.0).

    Attributes
    ----------
    x : ndarray
        Array of x-coordinates.
    y : ndarray
        Array of y-coordinates.
    matrix : ndarray
        Matrix of spline coefficients, shape (n, 4), where n is the number of intervals.
        Each row contains [a_i, b_i, c_i, d_i] for the i-th interval.
    bc_type : str
        Boundary condition type used for the spline.

    Methods
    -------
    __call__(xq)
        Evaluate the spline at the query points xq.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from functions.cb_spline import CubicSpline
    >>>
    >>> # Create some data points
    >>> x = np.linspace(0, 10, 10)
    >>> y = np.sin(x)
    >>>
    >>> # Create a cubic spline with natural boundary conditions
    >>> cs = CubicSpline(x, y)
    >>>
    >>> # Evaluate the spline at a finer grid
    >>> x_fine = np.linspace(0, 10, 100)
    >>> y_interp = cs(x_fine)
    >>>
    >>> # Plot the results
    >>> plt.figure(figsize=(10, 6))
    >>> plt.plot(x, y, 'o', label='Data points')
    >>> plt.plot(x_fine, y_interp, '-', label='Cubic spline')
    >>> plt.legend()
    >>> plt.title('Cubic Spline Interpolation')
    >>> plt.grid(True)
    >>> plt.show()
    >>>
    >>> # Example with clamped boundary conditions (specifying derivatives)
    >>> cs_clamped = CubicSpline(x, y, bc_type='clamped', bc=(0.0, 0.0))
    """

    def __init__(
        self,
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        bc_type: Literal[
            "natural", "clamped", "known", "parabolic", "extrapolated"
        ] = "natural",
        bc: Tuple[float, float] = (0.0, 0.0),
    ) -> None:
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self._h: npt.ArrayLike = np.diff(x)
        self.bc_type = bc_type.lower().strip()
        self.f1 = bc[0]
        self.f2 = bc[1]

        self._n = len(x) - 1  # number of intervals
        if self._n < 1:
            raise ValueError("Need at least two data points.")

        self._a = None
        self._b = None
        self._c = None
        self._d = None

        self._compute_coeffs()
        self.matrix = np.array([self._a, self._b, self._c, self._d]).T

    def _compute_coeffs(self) -> None:
        h = self._h
        B = self._build_B()
        A = self._build_A()
        c = self._solve_for_c(A, B)

        a = self.y[:-1]
        b = (self.y[1:] - self.y[:-1]) / h - h * (2 * c[:-1] + c[1:]) / 3
        d = (c[1:] - c[:-1]) / (3 * h)

        # Store internally
        self._a, self._b, self._c, self._d = a, b, c[:-1], d

    def _build_B(self) -> np.ndarray:
        n, h, y = self._n, self._h, self.y
        B = np.zeros(n - 1)

        if self.bc_type in ["natural", "parabolic", "extrapolated"]:
            for i in range(n - 1):
                B[i] = (
                    3 * (self.y[i + 2] - self.y[i + 1]) / h[i + 1]
                    - 3 * (self.y[i + 1] - self.y[i]) / h[i]
                )

        if self.bc_type in ["clamped", "known"]:
            for i in range(1, n - 1):
                B[i] = (
                    3 * (self.y[i + 2] - self.y[i + 1]) / h[i + 1]
                    - 3 * (self.y[i + 1] - self.y[i]) / h[i]
                )

        if self.bc_type == "natural":
            pass

        elif self.bc_type == "clamped":
            if self.f1 is None or self.f2 is None:
                raise ValueError("bc_type='clamped' needs f1, f2 (end derivatives).")
            B[0] = (
                3 * (self.y[2] - self.y[1]) / h[1]
                - 3 * (self.y[1] - self.y[0]) / h[0]
                - (3 / (2 * h[0])) * (self.y[1] - self.y[0])
                + 3 * self.f1 / 2
            )
            B[-1] = (
                3 * (self.y[n] - self.y[n - 1]) / h[n - 1]
                - 3 * (self.y[n - 1] - self.y[n - 2]) / h[n - 2]
                + 3 * (self.y[n] - self.y[n - 1]) / (2 * h[n - 1])
                - 3 * self.f2 / 2
            )

        elif self.bc_type == "extrapolated":
            # For extrapolated splines, no additional adjustments needed for B
            pass

        elif self.bc_type == "known":
            if self.f1 is None or self.f2 is None:
                raise ValueError("bc_type='known' (known curvature) needs f1, f2.")

            B[0] = (
                3.0 * ((self.y[2] - self.y[1]) / h[1] - (self.y[1] - self.y[0]) / h[0])
                - self.f1 * h[0] / 2.0
            )
            B[-1] = (
                3.0
                * (
                    (self.y[n] - self.y[n - 1]) / h[n - 1]
                    - (self.y[n - 1] - self.y[n - 2]) / h[n - 2]
                )
                - self.f2 * h[n - 1] / 2.0
            )

        elif self.bc_type == "parabolic":
            B[0] = 3.0 * (
                (self.y[2] - self.y[1]) / h[1] - (self.y[1] - self.y[0]) / h[0]
            )
            B[-1] = 3.0 * (
                (self.y[n] - self.y[n - 1]) / h[n - 1]
                - (self.y[n - 1] - self.y[n - 2]) / h[n - 2]
            )
        else:
            raise ValueError(f"Unknown bc_type '{self.bc_type}'")

        return np.array(B)

    def _build_A(self) -> np.ndarray:
        n = self._n
        h = self._h

        # Initialize main diagonal and off-diagonals
        D = [2 * (h[i] + h[i + 1]) for i in range(n - 1)]
        U = [h[i + 1] for i in range(n - 2)] if n > 2 else []
        L = [h[i + 1] for i in range(n - 2)] if n > 2 else []

        if self.bc_type == "clamped":
            D[0] = 2 * (h[0] + h[1]) - h[0] / 2
            D[-1] = 2 * (h[n - 2] + h[n - 1]) - h[n - 1] / 2

        elif self.bc_type == "extrapolated":
            # Correct the matrix for extrapolated boundary conditions
            D[0] = h[0] + h[0] ** 2 / h[1] + 2 * (h[0] + h[1])
            D[-1] = 2 * h[n - 2] + 3 * h[n - 1] + h[n - 1] ** 2 / h[n - 2]

            if n > 2:
                # Adjust first element of upper diagonal
                U = [h[i + 1] for i in range(n - 2)]
                U[0] = -h[0] ** 2 / h[1] + h[1]

                # Adjust last element of lower diagonal
                L = [h[i + 1] for i in range(n - 2)]
                L[-1] = h[n - 2] - h[n - 1] ** 2 / h[n - 2]

        elif self.bc_type == "parabolic":
            D[0] = 3 * h[0] + 2 * h[1]
            D[-1] = 3 * h[n - 1] + 2 * h[n - 2]

        # Build the tridiagonal matrix using scipy's diags
        diagonals = []
        offsets = []

        # Add main diagonal
        diagonals.append(D)
        offsets.append(0)

        # Add off-diagonals if they exist
        if L:
            diagonals.append(L)
            offsets.append(-1)
        if U:
            diagonals.append(U)
            offsets.append(1)

        A = diags(diagonals, offsets=offsets).toarray()
        return A

    def _solve_for_c(self, A, B) -> np.ndarray:
        h = self._h
        n = self._n
        # Solve the interior system for c[1..n-1]
        c_interior = np.linalg.solve(A, B) if n > 1 else np.array([0.0])

        if self.bc_type == "natural":
            c0, cN = 0.0, 0.0

        elif self.bc_type == "clamped":
            # c0, cN from your code snippet:
            c0 = (
                3 * (self.y[1] - self.y[0]) / (2 * h[0] ** 2)
                - 3 * self.f1 / (2 * h[0])
                - c_interior[0] / 2.0
            )
            cN = (
                -3 * (self.y[n] - self.y[n - 1]) / (2 * h[n - 1] ** 2)
                + 3 * self.f2 / (2 * h[n - 1])
                - c_interior[-1] / 2.0
            )

        elif self.bc_type == "extrapolated":
            c0 = c_interior[0] - (h[0] / h[1]) * (c_interior[1] - c_interior[0])
            cN = c_interior[n - 2] + (h[n - 1] / h[n - 2]) * (
                c_interior[n - 2] - c_interior[n - 3]
            )

        elif self.bc_type == "known":
            # c0 = 0.5*f1, cN = 0.5*f2
            if self.f1 is None or self.f2 is None:
                raise ValueError("bc_type='known' requires f1, f2.")
            c0 = 0.5 * self.f1
            cN = 0.5 * self.f2

        elif self.bc_type == "parabolic":
            # from your code, you effectively set c[0] = c_interior[0],
            # c[n] = c_interior[-1], or something similar.
            # We can do:
            c0 = c_interior[0]
            cN = c_interior[-1]

        else:
            c0 = 0.0
            cN = 0.0

        # Combine into single array
        c = np.concatenate(([c0], c_interior, [cN]))
        return c

    def __call__(self, xq: Union[float, npt.ArrayLike]) -> np.ndarray:
        xq = np.asarray(xq)
        i = np.clip(
            np.searchsorted(self.x, xq) - 1, 0, self._n - 1
        )  # Vectorized search
        dx = xq - self.x[i]
        return self._a[i] + self._b[i] * dx + self._c[i] * dx**2 + self._d[i] * dx**3
