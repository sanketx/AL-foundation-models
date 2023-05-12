"""Margins based sampling query strategy."""

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

from ALFM.src.clustering.kmeans import cluster_features
from ALFM.src.query_strategies.base_query import BaseQuery


class Margins(BaseQuery):
    """Selects samples with smallest difference in first and second predicted
    class softmax probabilities.
    """

    def __init__(
        self, enable_dropout: bool, cluster_features: bool, topk: float, **params: Any
    ) -> None:
        """Call the superclass constructor.

        Args:
            enable_dropout (bool): flag to enable dropout at inference.
        """
        super().__init__(**params)
        self.enable_dropout = enable_dropout
        self.cluster_features = cluster_features
        self.topk = topk

    def rank_features(self, probs: torch.Tensor) -> torch.Tensor:
        sorted_probs = probs.sort(dim=1)[0]
        margins = sorted_probs[:, -1] - sorted_probs[:, -2]
        return margins.argsort()

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

        features = self.features[unlabeled_indices]
        softmax_probs = self.model.get_probs(features, dropout=self.enable_dropout)
        indices = self.rank_features(softmax_probs)
        topk = round(self.topk * len(features))

        if self.cluster_features and topk > num_samples:
            vectors = torch.from_numpy(features[indices[:topk]])
            vectors = F.normalize(vectors).numpy()
            indices = cluster_features(vectors, num_samples)

        indices = indices[:num_samples]
        mask[unlabeled_indices[indices]] = True
        return mask
