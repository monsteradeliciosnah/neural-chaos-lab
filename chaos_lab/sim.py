import numpy as np


def lorenz(n=10000, dt=0.01, sigma=10.0, rho=28.0, beta=8 / 3):
    xs = np.zeros(n)
    ys = np.zeros(n)
    zs = np.zeros(n)
    xs[0] = 1.0
    ys[0] = 1.0
    zs[0] = 1.0
    for i in range(n - 1):
        x, y, z = xs[i], ys[i], zs[i]
        xs[i + 1] = x + dt * sigma * (y - x)
        ys[i + 1] = y + dt * (x * (rho - z) - y)
        zs[i + 1] = z + dt * (x * y - beta * z)
    return np.stack([xs, ys, zs], axis=1)


def henon(n=10000, a=1.4, b=0.3):
    x = np.zeros(n)
    y = np.zeros(n)
    x[0] = 0.1
    y[0] = 0.1
    for i in range(n - 1):
        x[i + 1] = 1 - a * x[i] ** 2 + y[i]
        y[i + 1] = b * x[i]
    return np.stack([x, y], axis=1)
