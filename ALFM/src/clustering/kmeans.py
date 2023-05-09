"""Implementation of K-Means++."""

import numpy as np
import torch
from faiss import pairwise_distances
from numpy.typing import NDArray
from rich.progress import track


def faiss_pd(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x, y = x.numpy(), y.numpy()
    dist_matrix = pairwise_distances(x, y)
    return torch.from_numpy(dist_matrix)


def torch_pd(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x, y = x.unsqueeze(0), y.unsqueeze(0)
    return torch.cdist(x, y).squeeze()


def kmeans_plus_plus_init(features: NDArray[np.float32], k: int) -> NDArray[np.int64]:
    centroids = []
    vectors = torch.from_numpy(features).cuda()
    n, d = vectors.shape

    # Choose the first centroid uniformly at random
    idx = np.random.randint(n)
    centroids.append(idx)

    # Compute the squared distance from all points to the centroid
    # pairwise_distances in FAISS returns the squared L2 distance
    centroid_vector = vectors[idx].view(1, -1)
    sq_dist = torch_pd(vectors, centroid_vector).ravel() ** 2
    sq_dist[centroids] = 0  # avoid numerical errors

    # Choose the remaining centroids
    for _ in track(range(1, k), description="[green]K-Means++ init"):
        probabilities = sq_dist / torch.sum(sq_dist)
        idx = torch.multinomial(probabilities, 1).item()  # type: ignore[assignment]
        centroids.append(idx)

        # update the squared distances
        centroid_vector = vectors[idx].view(1, -1)
        new_dist = torch_pd(vectors, centroid_vector).ravel() ** 2
        new_dist[centroids] = 0  # avoid numerical errors

        # update the minimum squared distance
        sq_dist = torch.minimum(sq_dist, new_dist)

    return np.array(centroids)
