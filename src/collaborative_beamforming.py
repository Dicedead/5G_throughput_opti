import numpy as np
import pycsou.abc.operator as pyop

from pycsou.opt.solver import PGD
from utils import L1NormMod, get_angle


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
    solver.fit(x0=np.ones(len(Q)))
    x = solver.solution()
    return x[:-1], x[-1]


def new_doas_from_density_matrix(density_matrix, base_station_positions, matrix_resolution=100, cluster_threshold=1):
    """
    Assign clusters closer to a station to that station and recompute more precise DOAs

    density_matrix : matrix with number of users in each pixel
    base_station_positions : list of positions of the base stations
    matrix_resolution : size of the side of one pixel in density_matrix
    cluster_threshold : Minimum number of users in a cluster in order to consider it
    """
    bs_doas = [[] for _ in range(len(base_station_positions))]

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

    return bs_doas


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
