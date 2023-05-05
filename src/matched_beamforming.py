import numpy as np


def matched_beam(p, lambda_, r0, N=10000):
    w = np.exp(-1j * 2 * np.pi * np.dot(p, r0) / lambda_)
    w = w / np.sqrt(np.sum(abs(w) ** 2))
    thetas = np.linspace(0, 2 * np.pi, N)
    r = np.array([np.cos(thetas), np.sin(thetas)])
    b_gain = abs(w @ np.exp(1j * 2 * np.pi * np.dot(p, r) / lambda_))
    return b_gain, thetas
