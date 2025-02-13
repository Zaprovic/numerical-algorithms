import numpy as np
import matplotlib.pyplot as plt
import numpy.typing as npt
from typing import Literal, Optional


def csextrapolated(x, y):
    n = len(x) - 1  # so we can have a set of points from x0,x1,x2,...,xn

    h = np.zeros(n)
    B = np.zeros(n - 1)

    for i in range(n):
        h[i] = x[i + 1] - x[i]

    for i in range(n - 1):
        B[i] = 3 * (y[i + 2] - y[i + 1]) / h[i + 1] - 3 * (y[i + 1] - y[i]) / h[i]

    D = np.diag([2 * (h[i] + h[i + 1]) for i in range(n - 1)])
    D[0, 0] = h[0] + h[0] ** 2 / h[1] + 2 * (h[0] + h[1])
    D[-1, -1] = 2 * h[n - 2] + 3 * h[n - 1] + h[n - 1] ** 2 / h[n - 2]

    U = np.diag([h[i] for i in range(1, n - 1)], k=1)
    U[0, 1] = -h[0] ** 2 / h[1] + h[1]

    L = np.diag([h[i] for i in range(1, n - 1)], k=-1)
    L[-1, 1] = h[n - 2] - h[n - 1] ** 2 / h[n - 2]

    A = D + L + U

    # constants
    a = np.array(y)
    c = np.linalg.solve(A, B)

    # conditions for extrapolated spline
    c0 = c[0] - (h[0] / h[1]) * (c[1] - c[0])
    cn = c[n - 2] + (h[n - 1] / h[n - 2]) * (c[n - 2] - c[n - 3])

    c = np.concatenate(([c0], c, [cn]))
    b = np.array(
        [(a[i + 1] - a[i]) / h[i] - h[i] * (2 * c[i] + c[i + 1]) / 3 for i in range(n)]
    )
    d = np.array([(c[i + 1] - c[i]) / (3 * h[i]) for i in range(n)])

    S = np.column_stack((a[0:-1], b, c[0:-1], d))

    return S


class CubicSpline:
    """
    Cubic spline interpolation with multiple possible boundary conditions:
        - natural
        - clamped (needs f1, f2)
        - extrapolated
        - known (curvature) (needs f1, f2)
        - parabolic

    Usage:
    ------
    >>> x = [0, 1, 2, 3]
    >>> y = [0, 1, 4, 9]
    >>> cs = CubicSpline(x, y, bc_type="natural")
    >>> cs._compute_coeffs()
    >>> # Evaluate spline at x = 1.5
    >>> print(cs(1.5))
    """

    def __init__(
        self,
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        bc_type: Literal[
            "natural", "clamped", "known", "parabolic", "extrapolated"
        ] = "natural",
        f1: Optional[float] = None,
        f2: Optional[float] = None,
    ):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self._h: npt.ArrayLike = np.diff(x)
        self.bc_type = bc_type.lower().strip()
        self.f1 = f1
        self.f2 = f2

        self._n = len(x) - 1  # number of intervals
        if self._n < 1:
            raise ValueError("Need at least two data points.")

        self._a = None
        self._b = None
        self._c = None
        self._d = None

        self.coefficients = self._compute_coeffs()

    def _compute_coeffs(self):
        h = self._h

        B = self._build_B()
        A = self._build_A()

        c = self._solve_for_c(A, B)

        a = self.y[:-1]
        b = np.zeros(self._n)
        d = np.zeros(self._n)

        for i in range(self._n):
            b[i] = (self.y[i + 1] - self.y[i]) / h[i] - (h[i] / 3.0) * (
                2 * c[i] + c[i + 1]
            )
            d[i] = (c[i + 1] - c[i]) / (3 * h[i])

        # Store internally
        self._a = a
        self._b = b
        self._c = c[:-1]  # note: c has length n+1, but interval i uses c[i]
        self._d = d

        return np.column_stack((d, c[:-1], b, a))

    def _build_B(self):
        n = self._n
        h = self._h
        B = np.zeros(n - 1)

        if self.bc_type in ["natural", "parabolic", "extrapolated"]:
            for i in range(n - 1):
                B[i] = (
                    3 * (self.y[i + 2] - self.y[i + 1]) / self._h[i + 1]
                    - 3 * (self.y[i + 1] - self.y[i]) / self._h[i]
                )

        if self.bc_type in ["clamped", "known"]:
            for i in range(1, n - 1):
                B[i] = (
                    3 * (self.y[i + 2] - self.y[i + 1]) / self._h[i + 1]
                    - 3 * (self.y[i + 1] - self.y[i]) / self._h[i]
                )

        if self.bc_type == "natural":
            pass

        elif self.bc_type == "clamped":
            if self.f1 is None or self.f2 is None:
                raise ValueError("bc_type='clamped' needs f1, f2 (end derivatives).")
            # B[0]
            B[0] = (
                3 * (self.y[2] - self.y[1]) / self._h[1]
                - 3 * (self.y[1] - self.y[0]) / self._h[0]
                - (3 / (2 * self._h[0])) * (self.y[1] - self.y[0])
                + 3 * self.f1 / 2
            )
            # B[-1]
            B[-1] = (
                3 * (self.y[n] - self.y[n - 1]) / self._h[n - 1]
                - 3 * (self.y[n - 1] - self.y[n - 2]) / self._h[n - 2]
                + 3 * (self.y[n] - self.y[n - 1]) / (2 * self._h[n - 1])
                - 3 * self.f2 / 2
            )

        elif self.bc_type == "extrapolated":
            B[0] = 3.0 * (
                (self.y[2] - self.y[1]) / h[1] - (self.y[1] - self.y[0]) / h[0]
            )
            B[-1] = 3.0 * (
                (self.y[n] - self.y[n - 1]) / h[n - 1]
                - (self.y[n - 1] - self.y[n - 2]) / h[n - 2]
            )

        elif self.bc_type == "known":
            if self.f1 is None or self.f2 is None:
                raise ValueError("bc_type='known' (known curvature) needs f1, f2.")

            B[0] = (
                3.0
                * (
                    (self.y[2] - self.y[1]) / self._h[1]
                    - (self.y[1] - self.y[0]) / self._h[0]
                )
                - self.f1 * self._h[0] / 2.0
            )
            B[-1] = (
                3.0
                * (
                    (self.y[n] - self.y[n - 1]) / self._h[n - 1]
                    - (self.y[n - 1] - self.y[n - 2]) / self._h[n - 2]
                )
                - self.f2 * self._h[n - 1] / 2.0
            )

        elif self.bc_type == "parabolic":
            B[0] = 3.0 * (
                (self.y[2] - self.y[1]) / self._h[1]
                - (self.y[1] - self.y[0]) / self._h[0]
            )
            B[-1] = 3.0 * (
                (self.y[n] - self.y[n - 1]) / self._h[n - 1]
                - (self.y[n - 1] - self.y[n - 2]) / self._h[n - 2]
            )
        else:
            raise ValueError(f"Unknown bc_type '{self.bc_type}'")

        return np.array(B)

    def _build_A(self):
        n = self._n
        h = self._h

        D = np.diag([2 * (self._h[i] + self._h[i + 1]) for i in range(n - 1)])
        U = np.diag([self._h[i] for i in range(1, n - 1)], k=1)
        L = np.diag([self._h[i] for i in range(1, n - 1)], k=-1)

        if self.bc_type == "natural":
            pass

        elif self.bc_type == "clamped":
            D[0, 0] = 2 * (h[0] + h[1]) - h[0] / 2
            D[-1, -1] = 2 * (h[n - 2] + h[n - 1]) - h[n - 1] / 2

        elif self.bc_type == "extrapolated":
            D[0, 0] = h[0] + h[0] ** 2 / h[1] + 2 * (h[0] + h[1])
            D[-1, -1] = 2 * h[n - 2] + 3 * h[n - 1] + h[n - 1] ** 2 / h[n - 2]

            U[0, 1] = -h[0] ** 2 / h[1] + h[1]
            L[-1, 1] = h[n - 2] - h[n - 1] ** 2 / h[n - 2]

        elif self.bc_type == "known":
            pass

        elif self.bc_type == "parabolic":
            D[0, 0] = 3 * self._h[0] + 2 * self._h[1]
            D[-1, -1] = 3 * self._h[n - 1] + 2 * self._h[n - 2]

        A = U + D + L
        return A

    def _solve_for_c(self, A, B):
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

    def __call__(self, xq):
        if self._a is None or self._b is None or self._c is None or self._d is None:
            raise RuntimeError("Must call ._compute_coeffs() before evaluating.")

        xq = np.asarray(xq)
        yq = np.zeros_like(xq, dtype=float)

        for j, xval in enumerate(xq):
            # Find i such that x[i] <= xval <= x[i+1].
            # You can do a binary search or just np.searchsorted:
            i = np.searchsorted(self.x, xval) - 1
            if i < 0:
                i = 0
            if i > self._n - 1:
                i = self._n - 1
            dx = xval - self.x[i]
            yq[j] = (
                self._a[i]
                + self._b[i] * dx
                + self._c[i] * (dx**2)
                + self._d[i] * (dx**3)
            )

        return yq
