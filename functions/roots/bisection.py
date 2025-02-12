import numpy as np
import pandas as pd
from typing import Callable


def bisection(
    f: Callable[[float], float], a: float, b: float, delta: float
) -> pd.DataFrame:
    """
    Bisection method for finding roots of a function

    Function : bisection(f, a, b, delta)
    f : Function must be created as a Python function or using lambdas (make sure to use sympy instead of numpy to declare special functions such as exp, sin, etc)
    a : Lower bound of the interval
    b : Upper bound of the interval
    delta : Maximum error that satisfies f(pn) < epsilon
    """
    if f(a) * f(b) > 0:
        raise ValueError("There is no root in the interval")

    i = 0
    fa = f(a)
    n0 = int(np.floor(np.log2((b - a) / delta)))
    data = []

    while i <= n0:
        p = a + (b - a) / 2
        fp = f(p)

        data.append([a, b, p, fp, i + 1])

        if fp == 0 or (b - a) / 2 < delta:
            return pd.DataFrame(data, columns=["a", "b", "p", "f(p)", "Iteration"])

        i += 1

        if fa * fp > 0:
            a = p
            fa = fp
        else:
            b = p

    return f"Method failed after {n0} iterations"
