import matplotlib.pyplot as plt
import numpy as np


def csplot(x, y, S):
    S = np.fliplr(S)
    n = len(x) - 1
    plt.grid()

    for i in range(n):
        xx = np.linspace(x[i], x[i+1], 1500)
        yy = np.polyval(S[i, :], xx - x[i])

        plt.plot(xx, yy, linewidth=2.5)

    plt.scatter(x, y, c="k")
    plt.show()
