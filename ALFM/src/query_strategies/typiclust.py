"""Typiclust query strategy."""

import logging
from typing import Any

import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from rich.progress import track

from ALFM.src.clustering.kmeans import cluster_features
from ALFM.src.init_strategies.typiclust_init import TypiclustInit
from ALFM.src.query_strategies.base_query import BaseQuery


class Typiclust(BaseQuery):
    """Typiclust active learning strategy.

    As described in Hacohen et al, "Active Learning on a Budget:
    Opposite Strategies Suit High and Low Budgets" (https://arxiv.org/abs/2202.02794).
    Code adapted from https://github.com/avihu111/TypiClust.
    """

    def __init__(
        self, knn: int, min_size: int, max_clusters: int, **params: Any
    ) -> None:
        """Call the superclass constructor."""
        super().__init__(**params)
        self.knn = knn
        self.min_size = min_size
        self.max_clusters = max_clusters

        # check if typiclust init was used and copy the graphs
        init_sampler = params["init_sampler"]

        if isinstance(init_sampler, TypiclustInit):
            if init_sampler.max_clusters != max_clusters:
                logging.warning(
                    "Mismatch in Typiclust max clusters - init: "
                    + f"{init_sampler.max_clusters}, query: {max_clusters}"
                )
                return

            if hasattr(init_sampler, "clust_labels"):
                logging.info("Copying precomputed cluster indices")
                self.clust_labels = init_sampler.clust_labels

    def _typical_vec_id(self, features: NDArray[np.float32], knn: int) -> int:
        index = faiss.IndexFlatL2(features.shape[1])
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)

        gpu_index.add(features)
        distances = gpu_index.search(features, knn + 1)[0].mean(axis=1)
        return distances.argmin()

    def _select_points(
        self,
        features: NDArray[np.float32],
        clust_labels: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        labeled_pool = torch.from_numpy(self.labeled_pool)
        clust_labels = clust_labels.clone()
        cluster_ids, cluster_sizes = torch.unique(clust_labels, return_counts=True)

        num_clusters = len(cluster_ids)
        count = torch.bincount(clust_labels[labeled_pool], minlength=num_clusters)

        cluster_df = pd.DataFrame(
            dict(
                cluster_id=cluster_ids,
                cluster_size=cluster_sizes,
                existing_count=count,
            )
        ).sort_values(["existing_count", "cluster_size"], ascending=[True, False])
        cluster_df = cluster_df[cluster_df.cluster_size > self.min_size]

        clust_labels[labeled_pool] = -1  # mark these as seen
        num_clusters = len(cluster_df)  # update after removing small clusters
        selected = torch.zeros(len(features), dtype=torch.bool)

        for i in track(range(num_samples), description="[green]Typiclust query"):
            idx = cluster_df.cluster_id.values[i % num_clusters]
            indices = torch.nonzero(clust_labels == idx).flatten()
            vectors = features[indices]

            if len(vectors) > self.min_size:
                knn = min(self.knn, len(indices) // 2)
                vec_id = self._typical_vec_id(vectors, knn)

                selected[indices[vec_id]] = True
                clust_labels[indices[vec_id]] = -1

        return torch.nonzero(selected).flatten()

    def query(self, num_samples: int) -> NDArray[np.bool_]:
        """Select a new set of datapoints to be labeled.

        Args:
            num_samples (int): The number of samples to select.

        Returns:
            NDArray[np.bool_]: A boolean mask for the selected samples.
        """
        labeled_indices = np.flatnonzero(self.labeled_pool)
        unlabeled_indices = np.flatnonzero(~self.labeled_pool)

        if num_samples > len(unlabeled_indices):
            raise ValueError(
                f"num_samples ({num_samples}) is greater than unlabeled pool size ({len(unlabeled_indices)})"
            )

        num_clusters = min(num_samples + len(labeled_indices), self.max_clusters)

        # Typiclust performs clustering on a fixed embedding space.
        embeddings = torch.from_numpy(self.features)
        # uncomment the line below to cluster featueres from the classifier
        # embeddings = self.model.get_embedding(self.features)
        vectors = F.normalize(embeddings).numpy()  # L2 normalized embeddings

        if not hasattr(self, "clust_labels"):  # check if precomputed
            _, clust_labels = cluster_features(vectors, num_clusters)

            if num_clusters == self.max_clusters:  # save cluster labels
                self.clust_labels = clust_labels

        else:  # load precomputed cluster labels
            clust_labels = self.clust_labels

        selected = self._select_points(vectors, clust_labels, num_samples)
        mask = np.zeros(len(self.features), dtype=bool)
        mask[selected] = True
        return mask
