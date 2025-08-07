import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from functions.diff_equations.pvi import vector_ode as vd

# x1 = u[0]
# x2 = u[1]
# y1 = u[2]
# y2 = u[3]

F = lambda t, u: np.array(
    [
        u[1],
        np.cos(t) * u[2] - np.exp(-t) * u[3] + np.arctan(t),
        u[3],
        np.log(t**2 + 1) * u[1] + u[0] * u[3],
    ]
)
a = 1
b = 4
alpha = np.array([0.2, 0.3, -0.2, -0.3])
h = 0.2
n = int((b - a) / h)

t, Z = vd(F, a, b, alpha, n, "heun")


df = pd.DataFrame(
    {"t": t, "x(t)": Z[:, 0], "x'(t)": Z[:, 1], "y(t)": Z[:, 2], "y'(t)": Z[:, 3]}
)

print(df)
