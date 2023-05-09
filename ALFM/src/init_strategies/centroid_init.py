"""Centroid init query class."""

from typing import Any

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

from ALFM.src.clustering.kmeans import kmeans_plus_plus_init
from ALFM.src.init_strategies.base_init import BaseInit


class CentroidInit(BaseInit):
    """Select samples based on their typicality scores."""

    def __init__(self, **params: Any) -> None:
        """Intialize the class with the feature and label arrays.

        Args:
            features (NDArray[np.float32]): array of input features.
            labels (NDArray[np.int64]): 1D array of target labels.
        """
        super().__init__(**params)

    def _cluster_features(
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
        """Select the intial set of datapoints to be labeled.

        Args:
            num_samples (int): The number of samples to select.

        Returns:
            NDArray[np.bool_]: A boolean mask for the selected samples.
        """
        if num_samples > len(self.features):
            raise ValueError(
                f"num_samples ({num_samples}) is greater than dataset size ({len(self.features)})"
            )

        features = torch.from_numpy(self.features)
        vectors = F.normalize(features).numpy()
        selected = self._cluster_features(vectors, int(num_samples))

        mask = np.zeros(len(self.features), dtype=bool)
        mask[selected] = True
        return mask
