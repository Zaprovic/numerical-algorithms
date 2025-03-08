# Numerical Algorithms

A Python module providing implementations of various numerical algorithms for mathematical computation.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/numerical-algorithms.git

# Navigate to the directory
cd numerical-algorithms

# Install the package
pip install -e .
```

## Usage

```python
# Import the module
import numerical_algorithms as na

# Example: Using a root-finding algorithm
result = na.newton_method(lambda x: x**2 - 5, lambda x: 2*x, initial_guess=2, tolerance=1e-10)
print(f"Square root of 5 ≈ {result}")
```

## Available Algorithms

### Root Finding
- `newton_method(f, df, initial_guess, tolerance=1e-10, max_iterations=100)`: Newton-Raphson method
- `bisection_method(f, a, b, tolerance=1e-10, max_iterations=100)`: Bisection method
- `secant_method(f, x0, x1, tolerance=1e-10, max_iterations=100)`: Secant method

### Numerical Integration
- `trapezoidal_rule(f, a, b, n)`: Trapezoidal rule for numerical integration
- `simpson_rule(f, a, b, n)`: Simpson's rule for numerical integration

### Linear Algebra
- `gaussian_elimination(A, b)`: Solve a system of linear equations using Gaussian elimination
- `lu_decomposition(A)`: LU decomposition of a matrix

### Differential Equations
- `euler_method(f, y0, t0, tf, h)`: Euler's method for solving ODEs
- `runge_kutta4(f, y0, t0, tf, h)`: Fourth-order Runge-Kutta method for solving ODEs

## Examples

### Finding a Root Using Newton's Method
```python
import numerical_algorithms as na
import math

# Define the function and its derivative
f = lambda x: math.cos(x) - x
df = lambda x: -math.sin(x) - 1

# Find the root starting from initial guess x=0.5
root = na.newton_method(f, df, initial_guess=0.5)
print(f"Root of cos(x) - x = 0: {root}")
```

### Solving an ODE Using Runge-Kutta 4
```python
import numerical_algorithms as na
import matplotlib.pyplot as plt
import numpy as np

# Define the ODE: dy/dt = -y
f = lambda t, y: -y

# Initial condition: y(0) = 1
# Solve from t=0 to t=5 with step size h=0.1
t, y = na.runge_kutta4(f, y0=1, t0=0, tf=5, h=0.1)

# Plot the solution
plt.plot(t, y)
plt.plot(t, np.exp(-np.array(t)), 'r--')  # Analytical solution y = e^(-t)
plt.legend(['Numerical solution', 'Analytical solution'])
plt.xlabel('t')
plt.ylabel('y')
plt.title('Solution to dy/dt = -y, y(0) = 1')
plt.show()
```

## Requirements
- Python 3.7+
- NumPy
- Matplotlib (for plotting examples)

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
