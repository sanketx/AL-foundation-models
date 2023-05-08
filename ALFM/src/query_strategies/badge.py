"""BADGE sampling class."""

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from rich.progress import track

from ALFM.src.query_strategies.base_query import BaseQuery


class BADGE(BaseQuery):
    """BADGE Active Learning - https://arxiv.org/abs/1906.03671."""

    def __init__(self, **params: Any) -> None:
        """Call the superclass constructor."""
        super().__init__(**params)

    def _pairwise_distances(
        self,
        Z1: torch.Tensor,
        P1: torch.Tensor,
        Z2: torch.Tensor,
        P2: torch.Tensor,
    ) -> torch.Tensor:
        term1 = torch.sum(Z1**2, dim=1)[:, None] * torch.sum(P1**2, dim=1)[:, None]
        term4 = torch.sum(Z2**2, dim=1)[None, :] * torch.sum(P2**2, dim=1)[None, :]
        term2 = (Z1 @ Z2.T) * (P1 @ P2.T)

        frob_norm_squared_matrix = term1 - 2 * term2 + term4
        return frob_norm_squared_matrix

    def _select_samples(
        self,
        vectors: torch.Tensor,
        delta: torch.Tensor,
        num_samples: int,
    ) -> NDArray[np.int64]:
        centroids = []
        n, d = vectors.shape

        # Choose the first centroid uniformly at random
        idx = np.random.randint(n)
        centroids.append(idx)

        # Compute the squared distance from all points to the centroid
        centroid_vector = (vectors[idx][None, :], delta[idx][None, :])
        sq_dist = self._pairwise_distances(*centroid_vector, vectors, delta).ravel()
        sq_dist[centroids] = 0  # avoid numerical errors

        # Choose the remaining centroids
        for _ in track(range(1, num_samples), description="[green]Badge query"):
            probabilities = sq_dist / torch.sum(sq_dist)
            idx = torch.multinomial(probabilities, 1).item()  # type: ignore[assignment]
            centroids.append(idx)

            # compute the new squared distances
            centroid_vector = (vectors[idx][None, :], delta[idx][None, :])
            n_dist = self._pairwise_distances(*centroid_vector, vectors, delta).ravel()
            n_dist[centroids] = 0  # avoid numerical errors

            # update the minimum squared distance
            sq_dist = torch.minimum(sq_dist, n_dist)

        return np.array(centroids)

    def query(self, num_samples: int) -> NDArray[np.bool_]:
        """Select a new set of datapoints to be labeled.

        Args:
            num_samples (int): The number of samples to select.

        Returns:
            NDArray[np.bool_]: A boolean mask for the selected samples.
        """
        unlabeled_features = self.features[~self.labeled_pool]

        if num_samples > len(unlabeled_features):
            raise ValueError(
                f"num_samples ({num_samples}) is greater than unlabeled pool size ({len(unlabeled_features)})"
            )

        probs, vectors = self.model.get_probs_and_embedding(unlabeled_features)
        labels = probs.argmax(dim=1)
        one_hot = F.one_hot(labels, num_classes=probs.shape[1])
        delta = probs - one_hot

        centroids = self._select_samples(vectors, delta, num_samples)

        mask = np.zeros(len(self.features), dtype=bool)
        unlabeled_indices = np.flatnonzero(~self.labeled_pool)
        mask[unlabeled_indices[centroids]] = True
        return mask
