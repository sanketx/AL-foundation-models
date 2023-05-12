"""Dropout sampling class."""

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

from ALFM.src.clustering.kmeans import cluster_features
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
        return torch.nonzero(mismatch > self.num_iter // 2).flatten()

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
        embeddings = F.normalize(embeddings)
        y_star = probs.argmax(dim=1)

        candidates = self._get_candidates(features, y_star)
        selected = cluster_features(embeddings[candidates].numpy(), int(num_samples))

        mask = np.zeros(len(self.features), dtype=bool)
        mask[unlabeled_indices[candidates[selected]]] = True
        return mask
