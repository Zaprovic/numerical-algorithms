import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(precision=5, suppress=True)


def helper():
    return f"""Newton's Polynomial Interpolation.
    
Output: 
    D : Matrix array with the divided differences table.
    c : Coefficients of the Newton's polynomial.
    
Parameters: newtonPoly(x,y)
    
Parameters: newtonPoly(X, Y, plot=False, function=None)
    X, Y: Dataset to interpolate.
    plot: When True, it will display a plot with continuous graph of the corresponding Newton's polynomial with discrete points where it interpolates.
    function: When specified (either as Python or lambda function), it will display a comparison plot with the function, polynomial interpolation and discrete X, Y points.
    
"""


def newtonpoly(x, y):
    n = len(x)
    F = np.zeros((n, n))
    F[:, 0] = y

    for i in range(1, n):
        for j in range(1, i + 1):
            F[i, j] = (F[i, j - 1] - F[i - 1, j - 1]) / (x[i] - x[i - j])

    # P = np.zeros(n)

    # for i in range(n):
    #     P[i] = F[i, i]

    #     for j in range(i - 1, -1, -1):
    #         P[j] = F[j, j] + (x[i] - x[j]) * P[j + 1]

    C = [F[n - 1, n - 1]]

    for k in range(n - 2, -1, -1):
        C = np.convolve(C, [1, -x[k]])  # Multiply by (x - X[k])
        C[-1] += F[k, k]  # Add divided difference term

    return C, F
