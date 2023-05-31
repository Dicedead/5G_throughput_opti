import numpy as np
import scipy as sp
import pycsou.abc.operator as pyop
from flexibeam import flexibeam
from music import music_algorithm

from pycsou.opt.solver import PGD
from utils import L1NormMod, get_angle


def collaborative_flexibeam(
        antenna_positions,
        station_positions,
        side_of_region,
        side_resolution,
        cov_per_station,
        cov_of_stations,
        number_of_doas,
        wavelength,
        resolution_music=0.15,
        resolution_flex=1e-4,
        lambda_=0.5,
        cluster_thresh=0
):
    sqrt_n_q = side_of_region // side_resolution
    x, y = np.arange(sqrt_n_q), np.arange(sqrt_n_q)
    r = dstack_product(x, y) * side_resolution + (side_resolution / 2.) * np.array([1, 1])

    steering_matrices = []
    beamforming_weights = []
    for s in range(len(antenna_positions)):
        steering_matrices.append(np.exp(-2 * np.pi * 1j * (antenna_positions[s] @ r.T)))

        doas, widths, _, _, _ = music_algorithm(
            cov_per_station[s], antenna_positions[s], wavelength, number_of_doas, resolution=resolution_music
        )
        _, _, w_weights_s = flexibeam(antenna_positions[s] - station_positions[s], doas, widths, wavelength)
        beamforming_weights.append(w_weights_s)

    density_estimation, _ = lasso_optimization(
        steering_matrices, beamforming_weights, cov_of_stations, lambda_
    )

    doas_per_station, widths_per_station = new_doas_from_density_matrix(
        density_estimation.reshape((int(sqrt_n_q), int(sqrt_n_q))), station_positions,
        matrix_resolution=side_resolution, cluster_threshold=cluster_thresh
    )

    thetas = []
    for s in range(len(antenna_positions)):
        beamforming_weights[s], theta, _ = flexibeam(antenna_positions[s] - station_positions[s],
                                                     np.array(doas_per_station[s]), np.array(widths_per_station[s]),
                                                     lambda_=wavelength, resolution=resolution_flex
                                                     )
        thetas.append(theta)

    return beamforming_weights, thetas


def dstack_product(x, y):
    return np.dstack(np.meshgrid(x, y)).reshape(-1, 2)


def lasso_optimization(steering_matrices, beamforming_weights, covariance_matrix, lambda_=0.5):
    """
    Perform the LASSO optimization to recover the density of user clusters and noise level

    :param steering_matrices: of each base station
    :param beamforming_weights: of each base station
    :param covariance_matrix: covariance of the signals emitted by each station
    :param lambda_: regularization parameter
    :return: gridded density of user clusters, estimated noise level
    """
    steering_matrix = np.vstack(steering_matrices)

    beamforming_cols = []
    zero_vector_length = len(beamforming_weights[0])
    num_zero_vectors = len(beamforming_weights) - 1
    for idx, w in enumerate(beamforming_weights):
        tmp = [np.zeros(zero_vector_length) for _ in range(num_zero_vectors)]
        tmp.insert(idx, w)
        beamforming_cols.append(np.array(tmp).reshape(-1, 1))
    beamforming_matrix = np.hstack(beamforming_cols)

    C = np.conjugate(steering_matrix.T) @ beamforming_matrix
    Ch = np.conjugate(C).T
    D = C @ Ch
    C_R_Ch = np.real(C @ covariance_matrix @ Ch)
    D_had_Dt = D * D.T

    Q = np.vstack([
        np.hstack([2 * D_had_Dt, -4 * np.real(np.diagonal(D @ D)).reshape(-1, 1)]),
        np.hstack([np.zeros(len(D)), np.array([2 * (np.linalg.norm(Ch @ C) ** 2)])])
    ])
    assert len(D) + 1 == len(Q)

    c = -2 * np.real(np.vstack([np.diagonal(C_R_Ch).reshape(-1, 1), np.trace(C_R_Ch)])).flatten()

    Q_op = pyop.LinOp.from_array(Q)
    c_op = pyop.LinFunc.from_array(c)

    tmp = [1. for _ in range(len(D))]
    tmp.append(0.)

    data_fid = pyop.QuadraticFunc(shape=(1, len(Q)), Q=Q_op, c=c_op)
    l1_reg = lambda_ * L1NormMod(shape=(1, len(Q)))

    solver = PGD(data_fid, l1_reg)
    solver.fit(x0=np.ones(len(Q), dtype=np.float64) + 0.1 * np.random.randn(len(Q)), tau=1 / sp.linalg.svdvals(Q)[0])
    x = solver.solution()
    return x[:-1], x[-1]


def new_doas_from_density_matrix(
        density_matrix,
        base_station_positions,
        matrix_resolution=100,
        cluster_threshold=1
):
    """
    Assign clusters closer to a station to that station and recompute more precise DOAs

    :param density_matrix : matrix with number of users in each pixel
    :param base_station_positions : list of positions of the base stations
    :param matrix_resolution : size of the side of one pixel in density_matrix
    :param cluster_threshold : Minimum number of users in a cluster in order to consider it
    :return doas and associated widths
    """
    bs_doas = [[] for _ in range(len(base_station_positions))]
    widths = [[] for _ in range(len(base_station_positions))]

    nonzero_indices = list(zip(*density_matrix.nonzero()))
    for i, j in nonzero_indices:
        if density_matrix[i, j] > cluster_threshold:
            x_cluster, y_cluster = (
                i * matrix_resolution + matrix_resolution / 2, j * matrix_resolution + matrix_resolution / 2
            )
            min_dist = -1
            closest_bs = -1
            for k in range(len(base_station_positions)):
                x = base_station_positions[k, 0]
                y = base_station_positions[k, 1]
                dist = (x - x_cluster) ** 2 + (y - y_cluster) ** 2
                if dist < min_dist or min_dist < 0:
                    min_dist = dist
                    closest_bs = k

            closest_bs_position = base_station_positions[closest_bs]
            doa = get_angle(*(np.array([x_cluster, y_cluster]) - closest_bs_position))
            bs_doas[closest_bs].append(doa)
            widths[closest_bs].append(2 * 360 * np.arcsin(matrix_resolution / (np.sqrt(2) * min_dist)) / (2 * np.pi))

    return bs_doas, widths


if __name__ == "__main__":
    density_matrix = np.array([
        [1, 0, 0, 5],
        [2, 1, 0, 7]
    ])
    base_station_positions = np.array([
        [300, 400],
        [0, 100]
    ])
    print(new_doas_from_density_matrix(density_matrix, base_station_positions))
