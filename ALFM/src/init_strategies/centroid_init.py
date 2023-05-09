"""Centroid init query class."""

from typing import Any
from typing import Tuple

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from rich.progress import track

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
        self, features: NDArray[np.float32], k: int
    ) -> Tuple[NDArray[np.int64], NDArray[np.float32]]:
        kmeans = faiss.Kmeans(
            features.shape[1],
            k,
            niter=100,
            gpu=1,
            verbose=True,
            max_points_per_centroid=128000,
        )
        init_idx = kmeans_plus_plus_init(features, k)
        kmeans.train(features, init_centroids=features[init_idx])
        _, clust_labels = kmeans.index.search(features, 1)
        return clust_labels.ravel(), kmeans.centroids

    def _select_points(
        self,
        features: NDArray[np.float32],
        clust_labels: NDArray[np.int64],
        centroids: NDArray[np.float32],
    ) -> NDArray[np.int64]:
        num_clusters, num_features = centroids.shape
        selected = []

        for i in track(range(num_clusters), description="[green]Centroid init"):
            indices = np.flatnonzero(clust_labels == i)
            vectors = features[indices]

            index = faiss.IndexFlatL2(num_features)
            index.add(vectors)

            idx = index.search(centroids[i].reshape(1, -1), 1)[1].item()
            selected.append(indices[idx])

        return np.array(selected)

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

        clust_labels, centroids = self._cluster_features(vectors, int(num_samples))
        selected = self._select_points(vectors, clust_labels, centroids)

        mask = np.zeros(len(self.features), dtype=bool)
        mask[selected] = True
        return mask
