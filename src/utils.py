import math

import numpy as np
import scipy as sp

import pycsou.abc.operator as pyop
import pycsou.operator.func as pyfu
import pycsou.util.ptype as pyct
from flexibeam import flexibeam


class L1NormMod(pyop.ProxFunc):
    def __init__(self, shape: pyct.OpShape):
        super().__init__(shape)
        n_out, n_in = shape
        self.l1norm = pyfu.L1Norm(dim=n_in - 1)

    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        return np.vstack([self.l1norm(arr[:-1]), arr[-1]])

    def prox(self, arr: pyct.NDArray, tau: pyct.Real) -> pyct.NDArray:
        return np.vstack([self.l1norm.prox(arr[:-1], tau=tau).reshape(-1, 1),
                          arr[-1]]
                         ).flatten()

    def lipschitz(self, **kwargs) -> pyct.Real:
        return self.l1norm.lipschitz()


def get_angle(x, y):
    """
    Gets angle of a vector (x,y)
    """
    angle = math.atan2(y, x) * 180 / math.pi
    if angle < 0:
        angle += 360
    return angle


def cartesian_to_arg(cartesian_coords):
    return np.angle(cartesian_coords[0] + cartesian_coords[1] * 1j)

def generate_covariance_matrix(
        antenna_positions,
        user_positions,
        user_intensities=None,
        additional_degrees_of_freedom=0,
        noise_level=0.1
):
    if user_intensities is None:
        user_intensities = np.ones(len(user_positions))

    steering_matrix = np.exp(-2 * np.pi * 1j * (antenna_positions @ user_positions.T))
    res = steering_matrix @ np.diag(user_intensities) @ np.conj(steering_matrix).T

    return res + noise_level * sp.stats.wishart.rvs(
        df=len(res) + additional_degrees_of_freedom,
        scale=np.eye(len(res)),
        size=1
    )


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
    r_user = np.zeros(len(transmitter_positions) ** 2)
    i = 0
    for pos in transmitter_positions:
        b_gain, _ = beamforming_method(antenna_positions, wavelength, pos, resolution=resolution)
        for pos in transmitter_positions:
            angle = cartesian_to_arg(pos)
            if(angle < 0):
                angle = angle + 2*np.pi
            angle_res = int(int(1./resolution) * angle/(2*np.pi))
            b = b_gain[angle_res]
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


if __name__ == "__main__":
    antenna_positions = np.array([
        [1, 3],
        [4, 6],
        [6, 10]
    ])

    user_positions = np.array([
        [2, 6],
        [10, 5],
        [6, 7],
        [9, 10]
    ])

    print(generate_covariance_matrix(antenna_positions, user_positions))
