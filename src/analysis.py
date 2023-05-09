import numpy as np


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
