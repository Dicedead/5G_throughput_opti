import numpy as np


def matched_beam(antenna_positions, lambda_, steering_dirs, resolution=1e-4):
    """
    Matched beamforming implementation

    :param antenna_positions: N x 2 array of cartesian coordinates
    :param lambda_: wavelength
    :param steering_dirs: steering antenna direction
    :param resolution: spectral resolution
    :return: beam shape, and all DOAs considered
    """
    w = np.exp(-1j * 2 * np.pi * np.dot(antenna_positions, steering_dirs) / lambda_)
    w = w / np.sqrt(np.sum(abs(w) ** 2))
    thetas = np.linspace(0, 2 * np.pi, int(1./resolution))
    r = np.array([np.cos(thetas), np.sin(thetas)])
    b = abs(w @ np.exp(1j * 2 * np.pi * np.dot(antenna_positions, r) / lambda_))
    return b, thetas
