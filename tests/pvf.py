import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functions.diff_equations.boundary_value import linear_shot as ls

# definimos las funciones p(x), q(x) y r(x)
p = lambda x: np.exp(-x) * np.sin(6 * x)
q = lambda x: np.exp(-x) * np.arctan(x)
r = lambda x: np.exp(-x) * np.cos(3 * x)

# definimos las cotas del intervalo
a = 6
b = 7

# definimos las condiciones de frontera
alpha = 2
beta = 4

h = 0.1
n = int((b - a) / h)


t, w = ls(p, q, r, a, b, alpha, beta, n)


df = pd.DataFrame({"x": t, "y(x)": w})
print(df)
