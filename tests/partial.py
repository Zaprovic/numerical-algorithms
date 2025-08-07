import os
import sys
from typing import Callable

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from functions.partial.elliptic import findiff as fd

f: Callable[[float, float], float] = lambda x, y: 6 * (x + y)

g1: Callable[[float], float] = lambda y: y**3 + 1
g2: Callable[[float], float] = lambda y: y**3 + 8

h1: Callable[[float], float] = lambda x: x**3 + 8
h2: Callable[[float], float] = lambda x: x**3 + 64

xa = 1
xb = 2

ya = 2
yb = 4

h = 0.01
k = 0.5

R = fd(f, g1, g2, h1, h2, xa, xb, ya, yb, h, k)

# Convert the result to a DataFrame for better visualization
df = pd.DataFrame(
    R,
    columns=pd.Index([f"x{i}" for i in range(R.shape[1])]),
    index=pd.Index([f"y{i}" for i in range(R.shape[0])]),
)

print(df)
