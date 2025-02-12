import numpy as np
from typing import Callable


def fixedpt(g: Callable[[float], float], p0: float, tol: float, m: int) -> str:
    """
    Fixed point iteration for finding roots of a function

    Function : fixedpt(g, p0, tol, m)
    g : Function must be created as a Python function or using lambdas (make sure to use sympy instead of numpy to declare special functions such as exp, sin, etc)
    p0 : Initial approximation of the root
    tol : Maximum error that satisfies f(pn) < epsilon
    m : Maximum number of iterations
    """

    P = [p0]

    for k in range(m + 1):
        P.append(g(P[k]))
        e = np.abs(P[k + 1] - P[k])
        p = P[k]

        if e < tol:
            break

    if k == m:
        return "Maximum number of iterations was exceeded"

    return f"Root: {p}\nIteration: {k+1}\nAbsolute error: {e}\nSequence: {P[1:]}"
