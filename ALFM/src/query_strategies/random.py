"""Random sampling class."""


import numpy as np
from ALFM.src.query_strategies.base_query import BaseQuery
from numpy.typing import NDArray


class Random(BaseQuery):
    """Randomly select a pool of samples to label."""

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

        indices = np.random.choice(unlabeled_indices, size=num_samples, replace=False)
        mask[indices] = True
        return mask
