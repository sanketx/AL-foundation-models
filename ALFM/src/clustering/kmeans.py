"""Implementation of K-Means++."""

from typing import Optional
from typing import Tuple

import faiss
import numpy as np
import torch
from numpy.typing import NDArray
from rich.progress import track


def faiss_pd(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x, y = x.numpy(), y.numpy()
    dist_matrix = faiss.pairwise_distances(x, y)
    return torch.from_numpy(dist_matrix)


def torch_pd(x: torch.Tensor, y: torch.Tensor, batch_size: int = 10240) -> torch.Tensor:
    result = torch.zeros(x.shape[0], y.shape[0])

    for i in range(0, x.shape[0], batch_size):
        for j in range(0, y.shape[0], batch_size):
            x_batch = x[i : i + batch_size]
            y_batch = y[j : j + batch_size]

            dists = torch.cdist(x_batch.unsqueeze(0), y_batch.unsqueeze(0)).squeeze(0)
            result[i : i + x_batch.shape[0], j : j + y_batch.shape[0]] = dists

    return result


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


def cluster_features(
    features: NDArray[np.float32],
    num_samples: int,
    weights: Optional[NDArray[np.float32]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_samples = int(num_samples)  # np int scalars cause problems with faiss

    kmeans = faiss.Kmeans(
        features.shape[1],
        num_samples,
        niter=100,
        gpu=1,
        verbose=True,
        min_points_per_centroid=1,
        max_points_per_centroid=512,
    )
    init_idx = kmeans_plus_plus_init(features, num_samples)
    kmeans.train(features, weights=weights, init_centroids=features[init_idx])

    sq_dist, cluster_idx = kmeans.index.search(features, 1)
    sq_dist = torch.from_numpy(sq_dist).ravel()
    cluster_idx = torch.from_numpy(cluster_idx).ravel()
    selected = torch.zeros(num_samples, dtype=torch.int64)

    for i in range(num_samples):
        idx = torch.nonzero(cluster_idx == i).ravel()
        min_idx = sq_dist[idx].argmin()  # point closest to the centroid
        selected[i] = idx[min_idx]  # add that id to the selected set

    return selected, cluster_idx
