import numpy as np


def flexibeam(antenna_positions, doas, widths, lambda_, resolution=1e-4):
    """
    Flexibeam implementation

    :param antenna_positions: N x 2 array of cartesian coordinates
    :param doas: np array of direction of arrivals, in degree
    :param widths: desired beamwidth in degree
    :param lambda_: wavelength
    :param resolution: spectral resolution
    :return beamshape and theta discretization
    """
    thetas = np.linspace(0, 2 * np.pi, int(1. / resolution))
    r = np.array([np.cos(thetas), np.sin(thetas)])

    angles = doas / 360 * 2 * np.pi
    r0 = np.array([np.cos(angles), np.sin(angles)])
    widths_rad = widths * 2 * np.pi / 360
    sigma = np.sqrt(2 - 2 * np.cos(widths_rad))

    ampli = np.exp(
        -2 * (np.pi ** 2) / (lambda_ ** 2) * np.dot((np.linalg.norm(antenna_positions, axis=1) ** 2).reshape((-1, 1)),
                                                  (sigma ** 2).reshape((1, -1)))
    )
    w = np.multiply(ampli, np.exp(-1j * 2 * np.pi * np.dot(antenna_positions, r0) / (lambda_)))
    w = w / np.sqrt(np.sum(abs(w) ** 2, axis=0))
    w = np.sum(w, axis=1)
    w = w / np.sqrt(np.sum(abs(w) ** 2))
    b_gain = abs((np.exp(1j * 2 * np.pi * np.dot(antenna_positions, r) / (lambda_))).T @ w)
    return b_gain, thetas
