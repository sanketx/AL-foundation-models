"""Typiclust query strategy."""

from typing import Any
from typing import Tuple

import numpy as np
import pandas as pd
from faiss import pairwise_distances
from numpy.typing import NDArray

from ALFM.src.query_strategies.base_query import BaseQuery


class Typiclust(BaseQuery):
    """Typiclust active learning strategy.

    As described in Hacohen et al, "Active Learning on a Budget:
    Opposite Strategies Suit High and Low Budgets" (https://arxiv.org/abs/2202.02794).
    Code aapted from https://github.com/avihu111/TypiClust.
    """

    MIN_CLUSTER_SIZE = 5
    MAX_NUM_CLUSTERS = 500
    K = 20

    def __init__(self, **params: Any) -> None:
        """Call the superclass constructor."""
        super().__init__(**params)

    def get_nn(
        self, features: NDArray[np.float32], k: int
    ) -> Tuple(NDArray[np.float32], NDArray[np.int64]):
        """Get the k nearest neighbors of each point in features."""
        distances = pairwise_distances(features, features)
        indices = np.argsort(distances, axis=1)[:, 1 : k + 1]
        distances = distances[np.arange(distances.shape[0])[:, None], indices]
        return distances, indices

    def get_mean_nn_dist(
        self, features: NDArray[np.float32], k: int, return_indices: bool = False
    ) -> NDArray[np.float32]:
        """Get the mean distance to the k nearest neighbors of each point in features."""
        distances, indices = self.get_nn(features, k)
        mean_distances = np.mean(distances, axis=1)
        if return_indices:
            return mean_distances, indices
        return mean_distances

    def compute_typicality(
        self, features: NDArray[np.float32], k: int
    ) -> NDArray[np.float32]:
        """Compute the typicality of each point in features."""
        mean_distances = self.get_mean_nn_dist(features, k)
        typicality = 1 / (mean_distances + 1e-5)
        return typicality

    def fast_kmeans(
        self,
        features: NDArray[np.float32],
        num_clusters: int,
    ):
        pass

    def init_clusters(
        self,
        num_samples: int,
    ):
        vectors = self.model.get_embedding(self.features)
        num_clusters = min(
            len(vectors[self.labeled_pool]) + num_samples, self.MAX_NUM_CLUSTERS
        )
        self.clusters = self.fast_kmeans(vectors, num_clusters)

    def query(self, num_samples: int) -> NDArray[np.bool_]:
        """Select a new set of datapoints to be labeled.

        Args:
            num_samples (int): The number of samples to select.

        Returns:
            NDArray[np.bool_]: A boolean mask for the selected samples.
        """
        if self.clusters is None:
            self.init_clusters(num_samples)

        mask = np.zeros(len(self.features), dtype=bool)
        labeled_indices = np.flatnonzero(self.labeled_pool)
        labels = np.copy(self.clusters)
        existing_indices = np.arange(len(labeled_indices))

        cluster_ids, cluster_sizes = np.unique(labels, return_counts=True)
        cluster_labeled_counts = np.bincount(
            labels[existing_indices], minlength=len(cluster_ids)
        )

        clusters_df = pd.DataFrame(
            {
                "cluster_id": cluster_ids,
                "cluster_size": cluster_sizes,
                "existing_count": cluster_labeled_counts,
                "neg_cluster_size": -1 * cluster_sizes,
            }
        )
        clusters_df = clusters_df[clusters_df.cluster_size > self.MIN_CLUSTER_SIZE]
        clusters_df = clusters_df.sort_values(["existing_count", "neg_cluster_size"])
        labels[existing_indices] = -1

        selected = []
        for i in range(num_samples):
            cluster = clusters_df.iloc[i % len(clusters_df)].cluster_id
            cluster_indices = np.flatnonzero(labels == cluster)
            feats = self.features[cluster_indices]
            typicality = self.compute_typicality(
                feats, min(self.K, len(cluster_indices) // 2)
            )
            idx = cluster_indices[np.argmax(typicality)]
            selected.append(idx)
            labels[idx] = -1

        selected = np.array(selected)
        assert (
            len(selected) == num_samples
        ), f"Expected {num_samples} samples, got {len(selected)}"
        assert (
            len(np.intersect1d(selected, existing_indices)) == 0
        ), "Selected samples must be unlabeled"
        mask[selected] = True
