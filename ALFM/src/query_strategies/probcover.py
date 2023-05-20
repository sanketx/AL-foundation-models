"""Implements ProbCover query strategy."""

import logging
from typing import Any
from typing import Optional
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from rich.progress import track

from ALFM.src.clustering.kmeans import cluster_features
from ALFM.src.clustering.kmeans import torch_pd
from ALFM.src.init_strategies.probcover_init import ProbcoverInit
from ALFM.src.query_strategies.base_query import BaseQuery


class ProbCover(BaseQuery):
    """ProbCover active learning query function.

    Selects samples that cover the unlabeled feature space according to the
    ProbCover algorithm as described in Yehuda et al. "Active Learning
    Through a Covering Lens" (https://arxiv.org/pdf/2205.11320.pdf).
    """

    def __init__(
        self, batch_size: int, delta_iter: int, delta: Optional[float], **params: Any
    ) -> None:
        """Call the superclass constructor."""
        super().__init__(**params)
        self.batch_size = batch_size
        self.delta_iter = delta_iter
        self.delta = delta

        # graph construction is deferred to the first query call
        # this preserves the random state for initial pool selection
        self.graph_constructed = False

        # check if probcover init was used and copy the graphs
        init_sampler = params["init_sampler"]

        if isinstance(init_sampler, ProbcoverInit):
            logging.info("Reusing existing graph from probcover init sampler")
            logging.info(f"overwriting delta: {self.delta} -> {init_sampler.delta}")
            self.delta = init_sampler.delta
            self.edge_list = init_sampler.edge_list
            self.graph_constructed = True

    def _build_graph(self) -> None:
        if not hasattr(self, "edge_list"):  # delta was calulated offline
            features, clust_labels = self._label_clusters()
            alpha = self._purity(self.delta, features.cuda(), clust_labels.cuda())
            logging.info(f"Graph built, delta: {self.delta}, alpha: {alpha}")

    def _estimate_delta(self, delta_iter: int) -> None:
        if self.delta is not None:
            logging.info(f"Delta: {self.delta} was computed offline")
            return

        features, clust_labels = self._label_clusters()
        self.delta, lower, upper = 0.5, 0.0, 1.0

        for i in range(delta_iter):
            alpha = self._purity(self.delta, features.cuda(), clust_labels.cuda())

            if alpha < 0.95:
                upper = self.delta
                self.delta = 0.5 * (lower + self.delta)

            else:
                lower = self.delta
                self.delta = 0.5 * (upper + self.delta)

            logging.info(f"iteration: {i}, delta: {self.delta}, alpha: {alpha}")

    def _label_clusters(self) -> Tuple[torch.Tensor, torch.Tensor]:
        features = torch.from_numpy(self.features)
        features = F.normalize(features)
        num_classes = len(np.unique(self.labels))

        _, clust_labels = cluster_features(features.numpy(), num_classes)
        return features, clust_labels

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

        for i in track(
            range(0, num_samples, step),
            description="[green]Pairwise distance calculation",
        ):
            fs = features[i : i + step]
            mask = torch_pd(fs, features, batch_size=len(features)) < delta
            nz_idx = torch.nonzero(mask)

            for j in range(len(fs)):
                neighbors = nz_idx[nz_idx[:, 0] == j][:, 1]
                match = clust_labels[i + j] == clust_labels[neighbors]
                count += match.all()

            nz_idx[:, 0] += i  # add batch offset
            edge_list.append(nz_idx)

        self.edge_list = torch.cat(edge_list)
        return count.item() / num_samples

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

        if not self.graph_constructed:  # first call to query
            self._estimate_delta(self.delta_iter)  # may need to be computed online
            self._build_graph()  # has to be computed on first call
            self.graph_constructed = True

            for idx in np.flatnonzero(self.labeled_pool):
                self._remove_covered(idx)

        for _ in track(range(num_samples), description="[green]Probcover query"):
            idx = self._highest_degree()
            self._remove_covered(idx)
            selected.append(idx)

        mask[selected] = True
        return mask
