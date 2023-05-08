"""Uncertainty sampling query strategy."""


from typing import Any

import numpy as np
from numpy.typing import NDArray

from ALFM.src.query_strategies.base_query import BaseQuery


class Uncertainty(BaseQuery):
    """Select samples with highest softmax uncertainty."""

    def __init__(self, enable_dropout: bool, **params: Any) -> None:
        """Call the superclass constructor.

        Args:
            enable_dropout (bool): flag to enable dropout at inference.
        """
        super().__init__(**params)
        self.enable_dropout = enable_dropout

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

        softmax_probs = self.model.get_probs(
            self.features[unlabeled_indices], dropout=self.enable_dropout
        )
        max_probs = softmax_probs.max(dim=1)[0]
        indices = max_probs.argsort()[:num_samples]
        mask[unlabeled_indices[indices]] = True
        return mask
