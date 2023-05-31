import math

import numpy as np
import pycsou.abc.operator as pyop
import pycsou.operator.func as pyfu
import pycsou.util.ptype as pyct


def get_angle(x, y):
    """
    Gets angle of a vector (x,y)
    """
    angle = math.atan2(y, x) * 180 / math.pi
    if angle < 0:
        angle += 360
    return angle


class L1NormMod(pyop.ProxFunc):
    def __init__(self, shape: pyct.OpShape):
        super().__init__(shape)
        n_out, n_in = shape
        self.l1norm = pyfu.PositiveL1Norm(dim=n_in - 1)

    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        return np.vstack([self.l1norm(arr[:-1]), arr[-1]])

    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        return np.vstack([self.l1norm.prox(arr[:-1], tau=tau).reshape(-1, 1), arr[-1].reshape(-1, 1)]).flatten()

    def lipschitz(self, **kwargs) -> pyct.Real:
        return self.l1norm.lipschitz()

def throughput_statistic(
        statistic,
        beamforming_method,
        antenna_positions,
        wavelength,
        transmitter_positions,
        channel_bandwidth_per_user=2e6,
        noise_level=0.1,
        c0=0.8,
        resolution=1e-4
):
    r_user = np.zeros(len(transmitter_positions) * int(1./resolution))
    i = 0
    for pos in transmitter_positions:
        b_gain, _ = beamforming_method(antenna_positions, wavelength, pos, resolution=resolution)
        for b in b_gain:
            r_user[i] = channel_bandwidth_per_user * np.log2(1 + c0 * (b ** 2) / noise_level)
            i += 1

    return statistic(r_user)


def average_throughput(
        beamforming_method,
        antenna_positions,
        wavelength,
        transmitter_positions,
        channel_bandwidth_per_user=2e6,
        noise_level=0.1,
        c0=0.8,
        resolution=1e-4
):
    return throughput_statistic(
        np.mean,
        beamforming_method,
        antenna_positions,
        wavelength,
        transmitter_positions,
        channel_bandwidth_per_user=channel_bandwidth_per_user,
        noise_level=noise_level,
        c0=c0,
        resolution=resolution
    )


def variance_throughput(
        beamforming_method,
        antenna_positions,
        wavelength,
        transmitter_positions,
        channel_bandwidth_per_user=2e6,
        noise_level=0.1,
        c0=0.8,
        resolution=1e-4
):
    return throughput_statistic(
        np.var,
        beamforming_method,
        antenna_positions,
        wavelength,
        transmitter_positions,
        channel_bandwidth_per_user=channel_bandwidth_per_user,
        noise_level=noise_level,
        c0=c0,
        resolution=resolution
    )
