import numpy as np


def forward(alpha, l, tn, N, M, f=None, g1=None, g2=None):
    """
    Parameters: Equation of the form ut(x,t) = alpha^2 uxx(x,t)
    Boundary conditions: u(x,0) = f(x) / u(0,t) = g1(t), u(l,t) = g2(t)
    Returns
    -------
    """

    # step sizes
    h = l / N
    k = tn / M
    lamda = alpha**2 * k / h**2

    D = np.diag([1-2*lamda for i in range(N+1)])
    U = np.diag([i+1 for i in range(N)], k=1)
    L = np.diag([i+1 for i in range(N)], k=-1)

    A = D + L + U

    return A


alpha = 3
l = 5
tn = 3
N = 5
M = 5

R = forward(alpha, l, tn, N, M)

print(R)
