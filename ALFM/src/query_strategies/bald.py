"""BALD query strategy."""

from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from ALFM.src.query_strategies.base_query import BaseQuery
from ALFM.src.query_strategies.entropy import Entropy


class BALD(BaseQuery):
    """The original BALD acquisition function.

    From Gal et al. "Deep Bayesian Active Learning with Image Data"
    (https://arxiv.org/abs/1703.02910).
    """

    def __init__(self, M: int, **params: Any) -> None:
        """Call the superclass constructor."""
        super().__init__(**params)
        self.M = M

    def _get_mc_samples(self, features: NDArray[np.float32]) -> torch.Tensor:
        """Get MC samples from the model.

        Returns:
            NDArray[np.float32]: MC samples from the model (M x N x C).
        """
        samples = torch.stack(
            [self.model.get_probs(features, dropout=True) for _ in range(self.M)]
        )
        return samples

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

        mc_samples = self._get_mc_samples(self.features[unlabeled_indices])

        H = Entropy.get_entropy(mc_samples.mean(dim=0))
        E = Entropy.get_entropy(mc_samples).mean(dim=0)
        mutual_information = H - E

        indices = mutual_information.argsort()[-num_samples:]
        mask[unlabeled_indices[indices]] = True
        return mask
