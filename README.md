# Numerical Algorithms

A Python library implementing various numerical algorithms for mathematical computations, scientific computing, and engineering applications.

## Overview

This project provides efficient implementations of common numerical algorithms used in computational mathematics. The `functions` module contains various numerical methods organized by category for solving different types of mathematical problems.

## Installation

Clone this repository:

```bash
git clone https://github.com/yourusername/numerical-algorithms.git
cd numerical-algorithms
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Module Structure

The `functions` module is organized into the following components:

### Root Finding (`roots.py`)

Methods for finding roots (zeros) of nonlinear equations:

- **`bisect(f, a, b, delta=1e-6)`**: Bisection method for finding roots in an interval

  - Repeatedly halves an interval and selects the subinterval where the root must lie
  - Parameters:
    - `f`: The function for which to find a root
    - `a`: Left endpoint of the initial interval
    - `b`: Right endpoint of the initial interval
    - `delta`: Desired accuracy (default: 1e-6)
  - Returns a DataFrame showing the steps of the algorithm

- **`fixedpt(g, p0, tol=1e-6, m=100)`**: Fixed-point iteration for finding roots

  - Uses the iteration $x_{n+1} = g(x_n)$ to converge to a fixed point
  - Parameters:
    - `g`: The function for which to find the fixed point
    - `p0`: Initial approximation
    - `tol`: Error tolerance (default: 1e-6)
    - `m`: Maximum iterations (default: 100)
  - Returns a DataFrame tracking the iteration history

- **`newton(f, x0, delta=1e-6, epsilon=1e-6, m=100)`**: Newton-Raphson method for fast convergence

  - Iteratively improves approximation using $x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}$
  - Parameters:
    - `f`: The function for which to find a root
    - `x0`: Initial approximation
    - `delta`: Tolerance for consecutive approximations (default: 1e-6)
    - `epsilon`: Tolerance for function value (default: 1e-6)
    - `m`: Maximum iterations (default: 100)
  - Returns the approximated root, error estimate, iteration count, and function value

- **`newtonMod(f, x0, delta=1e-6, epsilon=1e-6, m=100)`**: Modified Newton's method for multiple roots
  - Uses higher-order derivatives to improve convergence for multiple roots
  - Parameters:
    - `f`: The function for which to find a root
    - `x0`: Initial approximation
    - `delta`: Tolerance for consecutive approximations (default: 1e-6)
    - `epsilon`: Tolerance for function value (default: 1e-6)
    - `m`: Maximum iterations (default: 100)
  - Returns the approximated root, error estimate, iteration count, and function value
  - Particularly effective for multiple roots where standard Newton's method converges slowly

### Linear Systems (`linear.py`)

Methods for solving systems of linear equations using iterative approaches:

- **`jacobi(A, b, x0, epsilon=1e-6, max_iter=100)`**: Jacobi iterative method

  - Solves Ax = b using the formula: $\mathbf{x}^{(k+1)} = \mathbf{D}^{-1}(\mathbf{b} - (\mathbf{L}+\mathbf{U})\mathbf{x}^{(k)})$
  - Parameters:
    - `A`: Coefficient matrix (must be square with non-zero diagonal elements)
    - `b`: Right-hand side vector
    - `x0`: Initial guess for the solution vector
    - `epsilon`: Convergence tolerance (default: 1e-6)
    - `max_iter`: Maximum number of iterations (default: 100)
  - Returns the solution vector, number of iterations performed, and final relative error

- **`gseid(A, b, x0, epsilon=1e-6, max_iter=100)`**: Gauss-Seidel iterative method

  - Solves Ax = b using the formula: $\mathbf{x}^{(k+1)} = (\mathbf{D}+\mathbf{L})^{-1}(\mathbf{b} - \mathbf{U}\mathbf{x}^{(k)})$
  - Parameters:
    - `A`: Coefficient matrix (must be square with non-zero diagonal elements)
    - `b`: Right-hand side vector
    - `x0`: Initial guess for the solution vector
    - `epsilon`: Convergence tolerance (default: 1e-6)
    - `max_iter`: Maximum number of iterations (default: 100)
  - Returns the solution vector, number of iterations performed, and final relative error

- **`sor(A, b, x0, omega, epsilon=1e-6, max_iter=100)`**: Successive Over-Relaxation method
  - An accelerated version of the Gauss-Seidel method using a relaxation parameter
  - Parameters:
    - `A`: Coefficient matrix (must be square with non-zero diagonal elements)
    - `b`: Right-hand side vector
    - `x0`: Initial guess for the solution vector
    - `omega`: Relaxation parameter (0 < omega < 2)
    - `epsilon`: Convergence tolerance (default: 1e-6)
    - `max_iter`: Maximum number of iterations (default: 100)
  - Returns the solution vector, number of iterations performed, and final relative error
  - Special cases: omega=1 corresponds to Gauss-Seidel method

### Interpolation (`interp.py` and `cb_spline.py`)

Methods for interpolating data points:

- **`lagrange(x, y)`**: Lagrange polynomial interpolation

  - Computes the Lagrange polynomial interpolation and its coefficients
  - Parameters:
    - `x`: List of x-coordinates of the data points
    - `y`: List of y-coordinates of the data points
  - Returns a tuple containing:
    - The coefficients of the interpolation polynomial in descending order
    - The Lagrange basis polynomials

- **`newtonpoly(x, y)`**: Newton's divided differences method

  - Computes the Newton form of the interpolating polynomial using divided differences
  - Parameters:
    - `x`: 1-D array of x-coordinates of the points
    - `y`: 1-D array of y-coordinates of the points
  - Returns:
    - The coefficients of the Newton polynomial in descending order
    - The divided difference table
  - Especially useful for adding new points to an existing interpolation

- **`chebyshevNodes(a, b, n, f)`**: Generates Chebyshev nodes for optimal interpolation

  - Creates Chebyshev nodes in the interval [a, b] and computes function values at those nodes
  - Parameters:
    - `a`: Lower bound of the interval
    - `b`: Upper bound of the interval
    - `n`: Number of nodes
    - `f`: Function to compute at the nodes
  - Returns:
    - The Chebyshev nodes
    - Function values evaluated at the nodes
  - Helps minimize Runge's phenomenon in polynomial interpolation

- `CubicSpline`: Class implementing cubic spline interpolation with various boundary conditions

### Numerical Integration (`integration.py`)

Methods for numerical integration of functions:

- **`closed_newton_cotes(f, a, b, n)`**: Newton-Cotes formulas with endpoints

  - Approximates the definite integral using quadrature weights based on polynomial interpolation through equally spaced points including the endpoints
  - Parameters:
    - `f`: The function to integrate (must accept and return array input)
    - `a`: Lower limit of integration
    - `b`: Upper limit of integration
    - `n`: Number of subintervals to use
  - Returns the approximated value of the integral
  - Special cases: n=1 is the trapezoidal rule, n=2 is Simpson's rule

- **`open_newton_cotes(f, a, b, n)`**: Newton-Cotes formulas without endpoints

  - Approximates the definite integral using quadrature weights based on polynomial interpolation through equally spaced points excluding the endpoints
  - Parameters:
    - `f`: The function to integrate (must accept and return array input)
    - `a`: Lower limit of integration
    - `b`: Upper limit of integration
    - `n`: Number of interior points to use
  - Returns the approximated value of the integral
  - Useful for cases where function evaluation at endpoints is problematic

- **`closed_composite_newton_cotes(f, a, b, n, order=1)`**: Composite Newton-Cotes methods

  - Divides the integration interval into n subintervals and applies the closed Newton-Cotes formula of specified order to each subinterval
  - Parameters:
    - `f`: The function to integrate (must accept and return array input)
    - `a`: Lower limit of integration
    - `b`: Upper limit of integration
    - `n`: Number of subintervals to divide [a,b] into (must be divisible by the order)
    - `order`: Order of the Newton-Cotes formula to use in each subinterval (default is 1)
  - Returns the approximated value of the integral
  - Special cases: order=1 is the composite trapezoidal rule, order=2 is the composite Simpson's rule

- **`gaussLegendre(f, a, b, n)`**: Gaussian quadrature for high-accuracy integration
  - Uses Legendre polynomial roots and weights to approximate the integral
  - Parameters:
    - `f`: The function to integrate
    - `a`: Lower limit of integration
    - `b`: Upper limit of integration
    - `n`: Number of quadrature points
  - Returns the approximated value of the integral
  - Generally provides higher accuracy than Newton-Cotes formulas with the same number of function evaluations

### Differential Equations (`diff_equations/`)

- Initial Value Problems (`pvi.py`):
  - `scalar_ode()`: Methods for solving scalar ODEs
  - `vector_ode()`: Methods for solving systems of ODEs
  - Includes Euler, Modified Euler, Heun, Midpoint and RK4 methods
- Boundary Value Problems (`boundary_value.py`):
  - `linear_shot()`: Linear shooting method
  - `finite_differences()`: Finite difference method

### Partial Differential Equations (`partial/`)

- Parabolic PDEs (`parabolic.py`):

  - `forward()`: Explicit forward-difference method
  - `backward()`: Implicit backward-difference method
  - `crnich()`: Crank-Nicolson method

- Hyperbolic PDEs (`hyperbolic.py`):

  - `forward()`: Forward method for wave equations
  - `enhanced_forward()`: Enhanced stability forward method

- Elliptic PDEs (`elliptic.py`):
  - `findiff()`: Finite difference method for problems like Laplace/Poisson equations

## Usage Examples

### Root Finding

```python
import numpy as np
from functions.roots import bisect, newton

# Find root using bisection method
def f(x):
    return x**2 - 4

result = bisect(f, 1, 3, 1e-6)
print(f"Bisection method steps:\n{result}")

# Find root using Newton's method
root, error, iterations, value = newton(f, 3.0)
print(f"Root found: {root}, after {iterations} iterations")
```

### Numerical Integration

```python
import numpy as np
from functions.integration import gaussLegendre, closed_newton_cotes

# Integrate sin(x) from 0 to π
def f(x):
    return np.sin(x)

# Using Gaussian quadrature with 5 points
result_gauss = gaussLegendre(f, 0, np.pi, 5)
print(f"Gaussian quadrature result: {result_gauss}")

# Using Newton-Cotes formula
result_nc = closed_newton_cotes(f, 0, np.pi, 4)
print(f"Newton-Cotes result: {result_nc}")
```

### Differential Equations

```python
import numpy as np
from functions.diff_equations.pvi import scalar_ode

# Solve y' = -y with y(0) = 1 (solution is y = e^(-t))
def f(t, y):
    return -y

t, y = scalar_ode(f, 0, 5, 1.0, 100, "rk4")
print(f"Solution at t=5: {y[-1]}")  # Should be close to e^(-5)
```

### Cubic Spline Interpolation

```python
import numpy as np
import matplotlib.pyplot as plt
from functions.cb_spline import CubicSpline

# Create data points
x = np.linspace(0, 10, 10)
y = np.sin(x)

# Create cubic spline
cs = CubicSpline(x, y)

# Evaluate at finer points
x_fine = np.linspace(0, 10, 100)
y_interp = cs(x_fine)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'o', label='Data points')
plt.plot(x_fine, y_interp, '-', label='Cubic spline')
plt.legend()
plt.title('Cubic Spline Interpolation')
plt.grid(True)
plt.show()
```

### Partial Differential Equations

```python
import numpy as np
from functions.partial.parabolic import crnich

# Solve heat equation u_t = α²u_xx
def initial_temp(x):
    return np.sin(np.pi * x)

def boundary_left(t):
    return 0

def boundary_right(t):
    return 0

# Parameters
alpha = 0.5
x_range = (0, 1)
t_range = (0, 0.5)
h = 0.05  # spatial step
k = 0.01  # time step

solution = crnich(initial_temp, alpha, boundary_left, boundary_right,
  x_range[0], x_range[1], t_range[0], t_range[1], h, k, plot=True)
```

## Dependencies

- NumPy: For numerical computations
- SciPy: For sparse matrices and special functions
- Matplotlib: For visualization (optional)
- Pandas: For data organization and output

## Performance Considerations

The implementations focus on both clarity and efficiency. For large-scale problems, consider:

1. Using vectorized operations when possible
2. For very large systems, specialized libraries like SciPy sparse solvers
3. Adaptive step size methods for ODEs with rapidly changing solutions

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

- Burden, R. L., & Faires, J. D. (2010). Numerical Analysis (9th ed.). Brooks/Cole.
- Heath, M. T. (2018). Scientific Computing: An Introductory Survey (3rd ed.). SIAM.
- Press, W. H., Teukolsky, S. A., Vetterling, W. T., & Flannery, B. P. (2007). Numerical Recipes: The Art of Scientific Computing (3rd ed.). Cambridge University Press.
