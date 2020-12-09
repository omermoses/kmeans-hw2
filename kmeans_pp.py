import argparse
import pandas as pd
import numpy as np
import mykmeanssp as km


def k_mean(K, N, d, MAX_ITER, path):
    np.random.seed(0)
    observations_matrix = pd.read_csv(path, sep=',', header=None).to_numpy(dtype=np.float64)
    centroid_index_arr = np.full(K, -1, int)
    centroids_matrix=create_k_clusters(observations_matrix, N, K, d, centroid_index_arr)
    ind=[i for i in range(N) if observations_matrix[i] not in centroids_matrix]
    observations_matrix=(np.concatenate((centroids_matrix, observations_matrix[ind]), axis=0)).tolist()
    km.run([observations_matrix, K, N, d, MAX_ITER, centroid_index_arr.tolist()])


def create_k_clusters(observations_matrix, N, K, d, centroid_index_arr):
    centroids_matrix = np.full((K, d), 0, dtype=np.float64)
    first_centroid_index = np.random.choice(N, 1)
    # print(observations_arr)
    centroids_matrix[0] = observations_matrix[first_centroid_index[0]]
    centroid_index_arr[0] = first_centroid_index
    # print(centroids_matrix)
    find_next_centroids(observations_matrix, centroids_matrix, K, N, centroid_index_arr)
    return centroids_matrix


def find_next_centroids(observations_matrix, centroids_matrix, K, N, centroid_index_arr):
    i = 1  # already found one above

    while (i < K):
        # Run until we find k centroids
        min_d_arr = np.full(N, -1)
        for observation_index in range(N):
            # find ths distance for all observations
            min_d_arr[observation_index] = squared_euclidean_distance(observations_matrix[observation_index],
                                                                      centroids_matrix, i)
        min_d_arr = min_d_arr / (min_d_arr.sum())
        next_centroid_index = np.random.choice(N, 1, p=min_d_arr)
        centroid_index_arr[i] = next_centroid_index
        centroids_matrix[i] = observations_matrix[next_centroid_index]
        i += 1


def squared_euclidean_distance(observation, centroids_df, i):
    """find cluster’s centroid using squared Euclidean distance
    observation and centroid are lists of size D"""
    dist = np.power((centroids_df - observation), 2).sum(axis=1).min()
    return dist


if __name__ == '__main__':
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('K', action='store', type=int)
    my_parser.add_argument('N', action='store', type=int)
    my_parser.add_argument('d', action='store', type=int)
    my_parser.add_argument('MAX_ITER', action='store', type=int)
    my_parser.add_argument('filename', action='store', type=str)

    args = my_parser.parse_args()
    if args.K <= 0 or args.N <= 0 or args.d <= 0 or args.MAX_ITER <= 0:
        print("parameters should be greater then 0")
        exit(1)

    elif args.K >= args.N:
        print("K should be smaller then N")
        exit(1)

    # check that the path is valid

    k_mean(args.K, args.N, args.d, args.MAX_ITER, args.filename)
