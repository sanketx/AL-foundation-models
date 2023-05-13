"""Dropout sampling class."""

import logging
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
        self, features: NDArray[np.float32], y_star: torch.Tensor, num_samples: int
    ) -> torch.Tensor:
        samples = torch.stack(
            [
                self.model.get_probs(features, dropout=True).argmax(dim=1)
                for _ in range(self.num_iter)
            ]
        )

        thresh = self.num_iter // 2
        mismatch = (y_star != samples).sum(dim=0)

        while (mismatch > thresh).sum() < 25 * num_samples and thresh > 0:
            thresh = thresh - 1

        logging.info(
            f"Dropout iterations: {self.num_iter}, Mismatch threshold: {thresh}"
        )
        return torch.nonzero(mismatch > thresh).flatten()

    def _random_samples(
        self, candidates: torch.Tensor, num_samples: int
    ) -> torch.Tensor:
        num_unlabeled = np.count_nonzero(~self.labeled_pool)
        unlabeled_pool = torch.ones(num_unlabeled, dtype=torch.bool)
        unlabeled_pool[candidates] = False  # all candidates will be labeled

        remaining = torch.nonzero(unlabeled_pool).flatten()
        idx = np.random.choice(len(remaining), num_samples, replace=False)
        return remaining[idx]

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

        candidates = self._get_candidates(features, y_star, num_samples)

        if len(candidates) < num_samples:
            delta = num_samples - len(candidates)
            random_samples = self._random_samples(candidates, delta)
            candidates = torch.cat([candidates, random_samples])
            selected = torch.ones(len(candidates), dtype=torch.bool)

        else:
            candidate_vectors = F.normalize(embeddings[candidates]).numpy()
            selected, _ = cluster_features(candidate_vectors, num_samples)

        mask = np.zeros(len(self.features), dtype=bool)
        mask[unlabeled_indices[candidates[selected]]] = True
        return mask
