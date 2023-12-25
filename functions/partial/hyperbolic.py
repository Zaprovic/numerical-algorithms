import numpy as np

# hyperbolic differential equation
def forward(x0,xn,t0,tm,s,r,f,g,h,k):
    """
    Parameters
    ----------
    x0: Left x interval
    xn: Right x interval
    t0: Left time interval
    tm: Right time interval
    s: Boundary condition for u(x0,t) = s(t)
    r: Boundary condition for u(xn,t) = r(t)
    f: Boundary condition for u(x,t0) = f(x)
    g: Boundary condition for u(xn,t0) = g(x)
    h: Step size for x
    k: Step size for t

    Returns
    -------
    """

    N = round((xn-x0)/h)
    M = round((tm-x0)/k)


def enhanced_forward(x0,xn,t0,tm,s,r,f,g,h,k):
    """
    Parameters
    ----------
    x0: Left x interval
    xn: Right x interval
    t0: Left time interval
    tm: Right time interval
    s: Boundary condition for u(x0,t) = s(t)
    r: Boundary condition for u(xn,t) = r(t)
    f: Boundary condition for u(x,t0) = f(x)
    g: Boundary condition for u(xn,t0) = g(x)
    h: Step size for x
    k: Step size for t

    Returns
    -------
    """

    N = round((xn-x0)/h)
    M = round((tm-x0)/k)
