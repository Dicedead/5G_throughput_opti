import numpy as np
from scipy.signal import find_peaks, peak_prominences, peak_widths


def music_algorithm(cov_y, antenna_positions, lambda_, nbr_doas, resolution=0.1):
    """
    Implementation of the music algorithm for direction of arrival estimation

    :param cov_y: estimate of covariance matrix of antennas at one time t, (N x N)
    :param antenna_positions: N x 2 array of cartesian coordinates
    :param lambda_: wavelength
    :param nbr_doas: number of directional of arrivals to estimate
    :param resolution: spectral resolution
    :return: peak DOAs, width of beams, prominence of beams, complete estimated PSD, all DOAs considered
    """
    thetas = np.arange(0, 360, resolution)
    a = np.exp(
        -1j * 2 * np.pi / lambda_ * (antenna_positions @ [np.cos(thetas * np.pi / 180), np.sin(thetas * np.pi / 180)])
    ).reshape(antenna_positions.shape[0], -1)
    psd = np.zeros(thetas.shape)
    eigvals, eigvects = np.linalg.eigh(cov_y)
    u = eigvects[:, :len(antenna_positions) - nbr_doas]
    proj = u @ u.conj().T
    for i in range(len(thetas)):
        psd[i] = np.abs((a[:, i].conj().T @ a[:, i]) / (a[:, i].conj().T @ proj @ a[:, i]))
    peaks, _ = find_peaks(psd, height=np.mean(psd))
    widths = peak_widths(psd, peaks)[0] / len(psd) * 360
    prominences = peak_prominences(psd, peaks)[0]

    return thetas[peaks], widths, prominences, psd, thetas


