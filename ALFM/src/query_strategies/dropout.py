"""Dropout sampling class."""

from typing import Any

import faiss
import numpy as np
import torch
from numpy.typing import NDArray

from ALFM.src.clustering.kmeans import kmeans_plus_plus_init
from ALFM.src.query_strategies.base_query import BaseQuery


class Dropout(BaseQuery):
    """Uncertainty based on dropout inference consistency."""

    def __init__(self, num_iter: int, **params: Any) -> None:
        """Call the superclass constructor."""
        super().__init__(**params)
        self.num_iter = num_iter

    def _get_candidates(
        self, features: NDArray[np.float32], y_star: torch.Tensor
    ) -> torch.Tensor:
        samples = torch.stack(
            [
                self.model.get_probs(features, dropout=True).argmax(dim=1)
                for _ in range(self.num_iter)
            ]
        )

        mismatch = (y_star != samples).sum(dim=0)
        print(torch.unique(mismatch, return_counts=True))
        return torch.nonzero(mismatch > self.num_iter // 2).flatten()

    def _cluster_candidates(
        self, features: NDArray[np.float32], num_samples: int
    ) -> torch.Tensor:
        kmeans = faiss.Kmeans(
            features.shape[1],
            num_samples,
            niter=100,
            gpu=1,
            verbose=True,
            max_points_per_centroid=128000,
        )
        init_idx = kmeans_plus_plus_init(features, num_samples)
        kmeans.train(features, init_centroids=features[init_idx])

        sq_dist, cluster_idx = kmeans.index.search(features, 1)
        sq_dist = torch.from_numpy(sq_dist).ravel()
        cluster_idx = torch.from_numpy(cluster_idx).ravel()
        selected = torch.zeros(num_samples, dtype=torch.int64)

        for i in range(num_samples):
            idx = torch.nonzero(cluster_idx == i).ravel()
            min_idx = sq_dist[idx].argmin()  # point closest to the centroid
            selected[i] = idx[min_idx]  # add that id to the selected set

        return selected

    def query(self, num_samples: int) -> NDArray[np.bool_]:
        """Select a new set of datapoints to be labeled.

        Args:
            num_samples (int): The number of samples to select.

        Returns:
            NDArray[np.bool_]: A boolean mask for the selected samples.
        """
        unlabeled_indices = np.flatnonzero(~self.labeled_pool)

        if num_samples > len(unlabeled_indices):
            raise ValueError(
                f"num_samples ({num_samples}) is greater than unlabeled pool size ({len(unlabeled_indices)})"
            )

        features = self.features[unlabeled_indices]
        probs, embeddings = self.model.get_probs_and_embedding(features)
        y_star = probs.argmax(dim=1)

        candidates = self._get_candidates(features, y_star)
        selected = self._cluster_candidates(
            embeddings[candidates].numpy(), int(num_samples)
        )

        mask = np.zeros(len(self.features), dtype=bool)
        mask[unlabeled_indices[candidates[selected]]] = True
        return mask
