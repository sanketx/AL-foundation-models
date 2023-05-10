"""Implements ProbCover query strategy."""

import logging
import sys
from typing import Any

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from rich.progress import track

from ALFM.src.clustering.kmeans import kmeans_plus_plus_init
from ALFM.src.clustering.kmeans import torch_pd
from ALFM.src.query_strategies.base_query import BaseQuery


class ProbCover(BaseQuery):
    """ProbCover active learning query function.

    Selects samples that cover the unlabeled feature space according to the
    ProbCover algorithm as described in Yehuda et al. "Active Learning
    Through a Covering Lens" (https://arxiv.org/pdf/2205.11320.pdf).
    """

    def __init__(
        self, batch_size: int, delta_iter: int, delta: int, **params: Any
    ) -> None:
        """Call the superclass constructor."""
        super().__init__(**params)
        self.batch_size = batch_size
        self.delta = delta if delta is not None else self._estimate_delta(delta_iter)
        self._build_graph(delta)

        # remove edges of the labeled set
        self.labels_removed = False

    def _build_graph(self, delta: float) -> NDArray[np.int64]:
        if not hasattr(self, "edge_list"):  # delta was calulated onffline
            self._estimate_delta(1)

        print("graph built")

    def _estimate_delta(self, delta_iter: int) -> float:
        features = torch.from_numpy(self.features)
        features = F.normalize(features)
        num_classes = len(np.unique(self.labels))
        num_features = features.shape[1]

        clust_labels = self._cluster_features(features.numpy(), num_classes)
        clust_labels = torch.from_numpy(clust_labels).cuda()
        delta, lower, upper = 0.75, 0.5, 1.0

        for i in track(
            range(delta_iter), description="[green]Probcover delta estimation"
        ):
            alpha = self._purity(delta, features.cuda(), clust_labels)
            logging.info(f"iteration: {i}, delta: {delta}, alpha: {alpha}")

            if alpha < 0.95:
                upper = delta
                delta = 0.5 * (lower + delta)

            else:
                lower = delta
                delta = 0.5 * (upper + delta)

        return delta

    def _purity(
        self,
        delta: float,
        features: torch.Tensor,
        clust_labels: torch.Tensor,
    ) -> float:
        edge_list = []
        num_samples = len(features)
        count = torch.tensor(0, device="cuda")
        step = round(self.batch_size**2 / num_samples)

        for i in range(0, num_samples, step):
            fs = features[i : i + step]
            mask = torch_pd(fs, features) < delta
            nz_idx = torch.nonzero(mask)

            for j in range(len(fs)):
                neighbors = nz_idx[nz_idx[:, 0] == j][:, 1]
                match = clust_labels[i + j] == clust_labels[neighbors]
                count += match.all()

            nz_idx[:, 0] += i  # add batch offset
            edge_list.append(nz_idx)

        self.edge_list = torch.cat(edge_list)
        return count.item() / num_samples

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

    def _highest_degree(self) -> int:
        num_samples = len(self.features)
        counts = torch.bincount(self.edge_list[:, 0], minlength=num_samples)
        return counts.argmax().item()

    def _remove_covered(self, idx: int) -> None:
        covered = self.edge_list[self.edge_list[:, 0] == idx][:, 1]
        remove_idx = torch.isin(self.edge_list[:, 1], covered)
        self.edge_list = self.edge_list[~remove_idx]

    def query(self, num_samples: int) -> NDArray[np.bool_]:
        """Select a new set of datapoints to be labeled.

        Args:
            num_samples (int): The number of samples to select.

        Returns:
            NDArray[np.bool_]: A boolean mask for the selected samples.
        """
        mask = np.zeros(len(self.features), dtype=bool)
        unlabeled_indices = np.flatnonzero(~self.labeled_pool)

        if num_samples > len(unlabeled_indices):
            raise ValueError(
                f"num_samples ({num_samples}) is greater than unlabeled pool size ({len(unlabeled_indices)})"
            )

        selected = []

        if not self.labels_removed:
            for idx in np.flatnonzero(self.labeled_pool):
                self._remove_covered(idx)

            self.labels_removed = True

        for _ in track(range(num_samples), description="[green]Probcover query"):
            idx = self._highest_degree()
            self._remove_covered(idx)
            selected.append(idx)

        mask[selected] = True
        return mask
