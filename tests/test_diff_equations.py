import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("ggplot")

from functions.diff_equations.pvi import scalar_ode, vector_ode

f = lambda t, y: (y * np.cos(t) + 2 * t * np.exp(y)) / (
    np.sin(t) + t**2 * np.exp(y) + 2
)

F = lambda t, u: np.array(
    [
        u[1],
        np.log(t**2 + 4) - np.sin(u[1]) - np.exp(t) * u[0],
    ]
)

a = 3
b = 4
alpha = np.array([1, 0])
h = 0.05
n = int((b - a) / h)

t, U = vector_ode(F, a, b, alpha, n, "heun")

plt.scatter(t, U)
plt.show()
