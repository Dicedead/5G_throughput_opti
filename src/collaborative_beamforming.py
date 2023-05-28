import math

'''
Gets angle of a vector (x,y)
'''
def get_angle(x, y):
    angle = math.atan2(y, x) * 180 / math.pi
    if(angle < 0):
        angle += 360
    return angle


'''
density_matrix : matrix with number of users in each pixel
base_station_positions : list of positions of the base stations in a 5000x5000m (?) zone
matrix_resolution : size of the side of one pixel in density_matrix
cluster_threshold : Minimum number of users in a cluster in order to consider it 
'''
def new_doas_from_density_matrix(density_matrix, base_station_positions, matrix_resolution = 100, cluster_threshold = 5):
    bs_doas = [[] for i in range(len(base_station_positions))]

    for i in len(density_matrix):
        for j in len(density_matrix[i]):
            if(density_matrix[i][j] > cluster_threshold):
                x_cluster, y_cluster = (i*matrix_resolution + matrix_resolution/2, j*matrix_resolution + matrix_resolution/2)
                min_dist = -1
                closest_bs = -1
                for k in range(len(base_station_positions)):
                    x = base_station_positions[k][0]
                    y = base_station_positions[k][1]
                    dist = (x-x_cluster)**2 + (y-y_cluster)**2
                    if(dist < min_dist or min_dist < 0):
                        min_dist = dist
                        closest_bs = k
                
                closest_bs_position = base_station_positions[closest_bs]
                doa = get_angle((x_cluster, y_cluster) - closest_bs_position)
                bs_doas[closest_bs].append(doa)

    return bs_doas


