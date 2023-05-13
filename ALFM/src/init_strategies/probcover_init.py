"""Probcover init query class."""

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
from ALFM.src.init_strategies.base_init import BaseInit


class ProbcoverInit(BaseInit):
    """Select samples to maximize probability coverage."""

    def __init__(
        self, batch_size: int, delta_iter: int, delta: Optional[float], **params: Any
    ) -> None:
        """Call the superclass constructor."""
        super().__init__(**params)
        self.batch_size = batch_size

        if delta is None:  # the graph is built as part of the estimation process
            delta = self._estimate_delta(delta_iter)

        self.delta = delta
        self._build_graph()

    def _build_graph(self) -> None:
        if not hasattr(self, "edge_list"):  # delta was calulated offline
            features, clust_labels = self._label_clusters()
            alpha = self._purity(self.delta, features.cuda(), clust_labels.cuda())
            logging.info(f"Graph built, delta: {self.delta}, alpha: {alpha}")

    def _estimate_delta(self, delta_iter: int) -> float:
        features, clust_labels = self._label_clusters()
        delta, lower, upper = 0.5, 0.0, 1.0

        for i in track(range(delta_iter), description="[green]Delta estimation"):
            alpha = self._purity(delta, features.cuda(), clust_labels.cuda())

            if alpha < 0.95:
                upper = delta
                delta = 0.5 * (lower + delta)

            else:
                lower = delta
                delta = 0.5 * (upper + delta)

            logging.info(f"iteration: {i}, delta: {delta}, alpha: {alpha}")

        return delta

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

    def _highest_degree(self) -> int:
        num_samples = len(self.features)
        counts = torch.bincount(self.edge_list[:, 0], minlength=num_samples)
        return counts.argmax().item()

    def _remove_covered(self, idx: int) -> None:
        covered = self.edge_list[self.edge_list[:, 0] == idx][:, 1]
        remove_idx = torch.isin(self.edge_list[:, 1], covered)
        self.edge_list = self.edge_list[~remove_idx]

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

        selected = []

        for _ in track(range(num_samples), description="[green]Probcover Init query"):
            idx = self._highest_degree()
            self._remove_covered(idx)
            selected.append(idx)

        mask = np.zeros(len(self.features), dtype=bool)
        mask[selected] = True
        return mask
