"""Uncertainty sampling query strategy."""


import numpy as np
from ALFM.src.query_strategies.base_query import BaseQuery
from numpy.typing import NDArray


class Uncertainty(BaseQuery):
    """Select samples with highest softmax uncertainty."""

    def __init__(self) -> None:
        """Call the superclass constructor."""
        super().__init__()

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

        softmax_probs = self.model.get_probs(self.features[unlabeled_indices])
        max_probs = np.max(softmax_probs, axis=1)
        indices = np.argsort(max_probs)[:num_samples]
        mask[indices] = True
        return mask
