"""Balanced entropy acquisition strategy."""

from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from ALFM.src.query_strategies.base_query import BaseQuery
from ALFM.src.query_strategies.entropy import Entropy


class BalEntAcq(BaseQuery):
    """Balanced entropy query strategy.

    From Woo, "Active learning in Bayesian Neural Networks with Balanced
    Entropy Learning Principle" (https://openreview.net/pdf?id=ZTMuZ68B1g).
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

    def _differential_entropy(
        self, alpha: NDArray[np.float32], beta: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Calculate the differential entropy of the given softmax probabilities.

        Args:
            alpha (NDArray[np.float32]): The alpha parameters of the Beta distribution.
            beta (NDArray[np.float32]): The beta parameters of the Beta distribution.

        Returns:
            NDArray[np.float32]: The differential entropy.
        """
        alpha = torch.tensor(alpha)
        beta = torch.tensor(beta)

        def beta_func(x, y):
            return (
                torch.exp(torch.lgamma(x) + torch.lgamma(y) - torch.lgamma(x + y))
                + 1e-32
            )

        diff_entropy = (
            torch.log(beta_func(alpha + 1, beta))
            - alpha * torch.digamma(alpha + 1)
            - (beta - 1) * torch.digamma(beta)
            + (alpha + beta - 1) * torch.digamma(alpha + beta + 1)
        )

        return diff_entropy.numpy()

    def _marginalized_posterior_entropy(
        self, probs: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Calculate the marginalized posterior entropy of the given softmax probabilities.

        Let P ~ Beta(alpha, beta) be the probabilities. Then...
            E(P) := alpha / (alpha + beta) and
            Var(P) := alpha * beta / ((alpha + beta)^2 * (alpha + beta + 1)).
        Then solving for alpha and beta yields...
            alpha = E(P) * (1 - E(P)) / Var(P) - E(P) and beta = (1 / E(P) - 1) * alpha.

        MPE = sum(E(P) * h(P+)) + H(P)

        where h(P+) is the differential entropy of the conjugate Beta posterior entropy
        and H(P) is the Shannon entropy.

        Args:
            probs (NDArray[np.float32]): The probabilities.

        Returns:
            NDArray[np.float32]: The marginalized posterior entropy.
        """
        m = np.mean(probs, axis=0)  # E(P)
        sigma2 = np.var(probs, axis=0)  # Var(P)

        alpha = m * m * (1 - m) / sigma2 - m
        beta = (1 / m - 1) * alpha

        diff_entropy = self._differential_entropy(alpha, beta)  # h(P+)
        mp_entropy = np.sum(m * diff_entropy, axis=-1)
        mp_entropy += Entropy.get_entropy(np.mean(probs, axis=0))

        return mp_entropy

    def query(self, num_samples: int) -> NDArray[np.bool_]:
        """Select a new set of datapoints to be labeled.

        Balanced entropy principle:
            BalEnt = (marginal posterior entropy + H(Y)) / (H(Y) + log 2)
            If BalEnt >= 0, then BalEnt = 1 / BalEnt
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

        mp_entropy = self._marginalized_posterior_entropy(mc_samples)
        H = Entropy.get_entropy(np.mean(mc_samples, axis=0))

        balanced_entropy = (H + np.log(2)) / (mp_entropy + H)
        sign_idx = balanced_entropy < 0
        balanced_entropy[sign_idx] = 1 / balanced_entropy[sign_idx]

        indices = np.argsort(balanced_entropy)[-num_samples:]
        mask[unlabeled_indices[indices]] = True
        return mask
