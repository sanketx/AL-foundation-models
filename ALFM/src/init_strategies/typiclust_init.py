"""Typiclust init query class."""

from typing import Any

import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from rich.progress import track

from ALFM.src.clustering.kmeans import kmeans_plus_plus_init
from ALFM.src.init_strategies.base_init import BaseInit


class TypiclustInit(BaseInit):
    """Select samples based on their typicality scores."""

    def __init__(
        self, knn: int, min_size: int, max_clusters: int, **params: Any
    ) -> None:
        """Intialize the class with the feature and label arrays.

        Args:
            features (NDArray[np.float32]): array of input features.
            labels (NDArray[np.int64]): 1D array of target labels.
        """
        super().__init__(**params)
        self.knn = knn
        self.min_size = min_size
        self.max_clusters = max_clusters

    def _cluster_features(
        self, features: NDArray[np.float32], k: int
    ) -> NDArray[np.int64]:
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
        return clust_labels.ravel()

    def _typical_vec_id(self, features: NDArray[np.float32], knn: int) -> int:
        index = faiss.IndexFlatL2(features.shape[1])
        index.add(features)
        distances = index.search(features, knn + 1)[0].mean(axis=1)
        return distances.argmin()

    def _select_points(
        self,
        features: NDArray[np.float32],
        clust_labels: NDArray[np.int64],
        num_samples: int,
    ) -> NDArray[np.int64]:
        cluster_ids, cluster_sizes = np.unique(clust_labels, return_counts=True)
        num_clusters = len(cluster_ids)

        cluster_df = pd.DataFrame(
            dict(
                cluster_id=cluster_ids,
                cluster_size=cluster_sizes,
            )
        ).sort_values("cluster_size", ascending=False)
        cluster_df = cluster_df[cluster_df.cluster_size > self.min_size]

        num_clusters = len(cluster_df)  # update after removing small clusters
        selected = []

        for i in track(range(num_samples), description="[green]Typiclust init"):
            idx = cluster_df.cluster_id.values[i % num_clusters]
            indices = np.flatnonzero(clust_labels == idx)
            vectors = features[indices]

            knn = min(self.knn, len(indices) // 2)
            vec_id = self._typical_vec_id(vectors, knn)

            selected.append(indices[vec_id])
            clust_labels[indices[vec_id]] = -1

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

        num_clusters = min(num_samples, self.max_clusters)
        features = torch.from_numpy(self.features)
        vectors = F.normalize(features).numpy()

        clust_labels = self._cluster_features(vectors, int(num_clusters))
        selected = self._select_points(vectors, clust_labels, num_samples)

        mask = np.zeros(len(self.features), dtype=bool)
        mask[selected] = True
        return mask
