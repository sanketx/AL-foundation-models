"""PowerBALD query strategy."""

from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from ALFM.src.query_strategies.base_query import BaseQuery
from ALFM.src.query_strategies.entropy import Entropy


class PowerBALD(BaseQuery):
    """Modification to the BALD acquisition function as described in
    Kirsch et al. "Stochastic Batch Acquisition for Deep Active Learning"
    (https://arxiv.org/pdf/2106.12059.pdf).

    A simple change to the BALD query replacing top-K querying with a
    randomized variant.
    """

    def __init__(self, M: int, **params: Any) -> None:
        """Call the superclass constructor."""
        super().__init__(**params)
        self.M = M

    def _get_mc_samples(self, features: NDArray[np.float32]) -> torch.Tensor:
        """Get MC samples from the model.

        Returns:
            NDArray[np.float32]: MC samples from the model.
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
        s = H - E

        # sample the Gumbel distribution
        gumbel_samples = np.random.gumbel(0.0, 1.0, size=len(s))
        power_s = torch.log(s) + torch.from_numpy(gumbel_samples)

        indices = power_s.argsort()[-num_samples:]
        mask[unlabeled_indices[indices]] = True
        return mask
