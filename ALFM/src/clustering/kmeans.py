"""Implementation of K-Means++."""

import numpy as np
from faiss import pairwise_distances
from numpy.typing import NDArray
from rich.progress import track


def kmeans_plus_plus_init(vectors: NDArray[np.float32], k: int) -> NDArray[np.int64]:
    centroids = []
    vectors = vectors - vectors.mean(axis=0)  # to improve conditioning
    n, d = vectors.shape

    # Choose the first centroid uniformly at random
    first_centroid_idx = np.random.randint(n)
    centroids.append(first_centroid_idx)

    # Compute the squared distance from all points to the centroid
    # pairwise_distances in FAISS returns the squared L2 distance
    centroid_vector = vectors[centroids[-1]].reshape(1, -1)
    sq_dist = pairwise_distances(vectors, centroid_vector).ravel()
    sq_dist[first_centroid_idx] = 0

    # Choose the remaining centroids
    for _ in track(range(1, k), description="[green]K-Means++ init"):
        probabilities = sq_dist / np.sum(sq_dist)
        next_centroid_idx = np.random.choice(n, p=probabilities)
        centroids.append(next_centroid_idx)

        # update the squared distances
        centroid_vector = vectors[centroids[-1]].reshape(1, -1)
        new_dist = pairwise_distances(vectors, centroid_vector).ravel()
        sq_dist = np.minimum(sq_dist, new_dist)
        sq_dist[next_centroid_idx] = 0  # numerical stability for faiss

    return np.array(centroids)
