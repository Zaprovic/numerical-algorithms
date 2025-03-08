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
- `bisect()`: Bisection method for finding roots in an interval
- `fixedpt()`: Fixed-point iteration for finding roots
- `newton()`: Newton-Raphson method for fast convergence
- `newtonMod()`: Modified Newton's method for handling multiple roots

### Linear Systems (`linear.py`)
- `jacobi()`: Jacobi iterative method
- `gseid()`: Gauss-Seidel iterative method
- `sor()`: Successive Over-Relaxation method

### Interpolation (`interp.py` and `cb_spline.py`)
- `lagrange()`: Lagrange polynomial interpolation
- `newtonpoly()`: Newton's divided differences method
- `chebyshevNodes()`: Generates Chebyshev nodes for optimal interpolation
- `CubicSpline`: Class implementing cubic spline interpolation with various boundary conditions

### Numerical Integration (`integration.py`)
- `closed_newton_cotes()`: Newton-Cotes formulas with endpoints
- `open_newton_cotes()`: Newton-Cotes formulas without endpoints
- `closed_composite_newton_cotes()`: Composite Newton-Cotes methods
- `gaussLegendre()`: Gaussian quadrature for high-accuracy integration

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

## Acknowledgments

- Inspired by classic numerical methods textbooks
- Thanks to all contributors who have helped improve this library
