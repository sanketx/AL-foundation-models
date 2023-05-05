"""BALD query strategy."""

from typing import Any

import numpy as np
from ALFM.src.query_strategies.base_query import BaseQuery
from numpy.typing import NDArray


class BALD(BaseQuery):
    """The original BALD acquisition function.

    From Gal et al. "Deep Bayesian Active Learning with Image Data"
    (https://arxiv.org/abs/1703.02910).
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
