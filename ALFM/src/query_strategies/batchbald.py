"""BatchBALD query strategy."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ALFM.src.query_strategies.base_query import BaseQuery


class BatchBALD(BaseQuery):
    """BatchBALD query strategy.

    Selects a batch of samples with highest mutual information between the
    samples and the model parameters. Described in Kirsh et al., "BatchBALD:
    Efficient and Diverse Batch Acquisition for Deep Bayesian Active Learning"
    (https://arxiv.org/abs/1906.08158).
    """

    def __init__(self, **params: Any) -> None:
        """Call the superclass constructor."""
        super().__init__(**params)

    def query(self, num_samples: int) -> NDArray[np.bool_]:
        """Select a new set of datapoints to be labeled.

        Args:
            num_samples (int): The number of samples to select.

        Returns:
            NDArray[np.bool_]: A boolean mask for the selected samples.
        """
        pass
