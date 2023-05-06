"""PowerBALD query strategy."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ALFM.src.query_strategies.base_query import BaseQuery


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

    def _get_mc_samples(self, features: NDArray[np.float32]) -> NDArray[np.float32]:
        """Get MC samples from the model.

        Returns:
            NDArray[np.float32]: MC samples from the model.
        """
        samples = np.stack(
            [self.model.get_probs(features, dropout=True) for _ in range(self.M)]
        )
        return samples

    def _shannon_entropy(self, probs: NDArray[np.float32]) -> NDArray[np.float32]:
        """Calculate the Shannon entropy of the given softmax probabilities.

        Args:
            probs (NDArray[np.float32]): The probabilities.

        Returns:
            NDArray[np.float32]: The Shannon entropy.
        """
        return -np.sum(probs * np.log(probs + 1e-10), axis=-1)

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

        H = self._shannon_entropy(np.mean(mc_samples, axis=0))
        E = np.mean(self._shannon_entropy(mc_samples), axis=0)
        s = H - E

        # sample the Gumbel distribution
        gumbel_samples = np.random.gumbel(0.0, 1.0, size=len(s))
        power_s = np.log(s) + gumbel_samples

        indices = np.argsort(power_s)[-num_samples:]
        mask[unlabeled_indices[indices]] = True
        return mask
