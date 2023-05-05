"""Balanced entropy acquisition strategy."""

from typing import Any

import numpy as np
from ALFM.src.query_strategies.base_query import BaseQuery
from numpy.typing import NDArray


class BalEntAcq(BaseQuery):
    """Balanced entropy query strategy.

    From Woo, "Active learning in Bayesian Neural Networks with Balanced
    Entropy Learning Principle" (https://openreview.net/pdf?id=ZTMuZ68B1g).
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
