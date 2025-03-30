import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from functions.roots import bisect, fixedpt, newton, newtonMod


class TestBisectionMethod:
    def test_simple_root(self):
        """Test bisection method with a function that has a simple root."""
        f = lambda x: x**2 - 4  # Root at x = 2
        result = bisect(f, 1, 3)

        assert isinstance(result, pd.DataFrame)
        assert abs(result["p"].iloc[-1] - 2.0) < 1e-6
        assert abs(result["f(p)"].iloc[-1]) < 1e-6

    def test_exact_root(self):
        """Test when bisection hits an exact root."""
        f = lambda x: x**3 - x - 2  # Root at x ≈ 1.521
        result = bisect(f, 1, 2)

        assert isinstance(result, pd.DataFrame)
        assert abs(result["p"].iloc[-1] - 1.521) < 1e-3

    def test_error_no_root(self):
        """Test error when no root exists in interval."""
        f = lambda x: x**2 + 1  # No real roots
        with pytest.raises(ValueError, match="There is no root in the interval"):
            bisect(f, -1, 1)


class TestFixedPointMethod:
    def test_convergent_function(self):
        """Test fixed point iteration with a convergent function."""
        g = lambda x: np.exp(-x)  # Fixed point at x ≈ 0.5671
        result = fixedpt(g, 0.5)

        assert isinstance(result, pd.DataFrame)
        assert abs(result["p"].iloc[-1] - 0.5671) < 1e-3

    def test_with_different_initial_guess(self):
        """Test fixed point with a different initial guess."""
        g = lambda x: np.cos(x)  # Fixed point at x ≈ 0.7391
        result = fixedpt(g, 1.0)

        assert isinstance(result, pd.DataFrame)
        assert abs(result["p"].iloc[-1] - 0.7391) < 1e-3

    def test_max_iterations_exceeded(self):
        """Test error when maximum iterations exceeded."""
        g = lambda x: 2 * x  # Divergent function
        with pytest.raises(
            ValueError, match="Maximum number of iterations was exceeded"
        ):
            fixedpt(g, 1.0, tol=1e-6, m=10)

    def test_linear_convergent(self):
        """Test with a linear convergent function."""
        # Uses g(x) = (x+2)/2 which converges to fixed point x=2
        g = lambda x: (x + 2) / 2
        result = fixedpt(g, 1.0)

        assert isinstance(result, pd.DataFrame)
        assert abs(result["p"].iloc[-1] - 2.0) < 1e-6


class TestNewtonMethod:
    def test_simple_root(self):
        """Test Newton's method with a simple function."""
        f = lambda x: x**2 - 4  # Root at x = 2 or x = -2
        root, error, iterations, f_value = newton(f, 3.0)

        assert abs(root - 2.0) < 1e-6
        assert abs(f_value) < 1e-6

    def test_cubic_function(self):
        """Test Newton's method with a cubic function."""
        f = lambda x: x**3 - x - 2  # Root at x ≈ 1.521
        root, error, iterations, f_value = newton(f, 2.0)

        assert abs(root - 1.521) < 1e-3
        assert abs(f_value) < 1e-6

    def test_derivative_zero(self):
        """Test error when derivative is zero."""
        f = lambda x: x**2  # Derivative is zero at x = 0
        with pytest.raises(ZeroDivisionError):
            newton(f, 0.0)


class TestModifiedNewtonMethod:
    def test_simple_root(self):
        """Test modified Newton's method with a simple function."""
        f = lambda x: x**2 - 4  # Root at x = 2 or x = -2
        root, error, iterations, f_value = newtonMod(f, 3.0)

        assert abs(root - 2.0) < 1e-6
        assert abs(f_value) < 1e-6

    def test_multiple_root(self):
        """Test with a function that has a multiple root."""
        f = lambda x: x**2  # Double root at x = 0
        root, error, iterations, f_value = newtonMod(f, 0.5)

        # Using a slightly more forgiving tolerance for the root
        assert abs(root) < 1e-5
        assert abs(f_value) < 1e-6


if __name__ == "__main__":
    pytest.main()
