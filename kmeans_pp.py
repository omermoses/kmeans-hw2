import argparse
import pandas as pd
import numpy as np

def k_mean(K, N, d, MAX_ITER, path):
    np.random.seed(0)
    print("")
    observations_df = pd.read_csv(path, sep=',', header=None)
    create_k_clusters(observations_df, N, K)


def create_k_clusters(observations_df, N, K):
    centroids_matrix = pd.DataFrame()
    first_centroid_index = np.random.choice(N, 1)
    print(observations_df.iloc[first_centroid_index[0]])
    centroids_matrix[0] = (observations_df.iloc[first_centroid_index[0]]).T
    print(centroids_matrix)
    find_next_centroid(observations_df, centroids_matrix, K)



def find_next_centroid(observations_df, centroids_matrix, K):

    i = 1

    while (i<K):
        #Run until we find k centroids
        for observation_index in observations_df.index:
            squared_euclidean_distance(observations_df[observation_index], centroids_matrix, i)

        i+=1

def squared_euclidean_distance(observation, centroids_matrix, i):
    """find cluster’s centroid using squared Euclidean distance
    observation and centroid are lists of size D"""
    dist= 0
    for centroid_index in centroids_matrix.index:
        for index in range(len(observation)):
            dist += (observation[index] - centroids_matrix[centroid_index][index]) ** 2
    return dist

if __name__ == '__main__':
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('K', action='store', type=int)
    my_parser.add_argument('N', action='store', type=int)
    my_parser.add_argument('d', action='store', type=int)
    my_parser.add_argument('MAX_ITER', action='store', type=int)
    my_parser.add_argument('filename', action='store', type=str)

    args = my_parser.parse_args()



    k_mean(args.K, args.N, args.d, args.MAX_ITER, args.filename)
