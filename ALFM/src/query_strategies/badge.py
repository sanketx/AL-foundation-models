"""BADGE sampling class."""

from typing import Any

import numpy as np
from ALFM.src.query_strategies.base_query import BaseQuery
from numpy.typing import NDArray
from rich.progress import track


class BADGE(BaseQuery):
    """BADGE Active Learning - https://arxiv.org/abs/1906.03671."""

    def __init__(self, **params: Any) -> None:
        """Call the superclass constructor."""
        super().__init__(**params)

    def _pairwise_distances(
        self,
        Z1: NDArray[np.float32],
        P1: NDArray[np.float32],
        Z2: NDArray[np.float32],
        P2: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        term1 = np.sum(Z1**2, axis=1)[:, None] * np.sum(P1**2, axis=1)[:, None]
        term4 = np.sum(Z2**2, axis=1)[None, :] * np.sum(P2**2, axis=1)[None, :]
        term2 = (Z1 @ Z2.T) * (P1 @ P2.T)

        frob_norm_squared_matrix = term1 - 2 * term2 + term4
        return frob_norm_squared_matrix  # type: ignore[no-any-return]

    def _select_samples(
        self,
        vectors: NDArray[np.float32],
        delta: NDArray[np.float32],
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
        sq_dist[idx] = 0  # avoid numerical errors

        # Choose the remaining centroids
        for _ in track(range(1, num_samples), description="[green]Badge query"):
            probabilities = sq_dist / np.sum(sq_dist)
            idx = np.random.choice(n, p=probabilities)
            centroids.append(idx)

            # update the squared distances
            centroid_vector = (vectors[idx][None, :], delta[idx][None, :])
            n_dist = self._pairwise_distances(*centroid_vector, vectors, delta).ravel()
            sq_dist = np.minimum(sq_dist, n_dist)
            sq_dist[idx] = 0  # avoid numerical errors

        return np.array(centroids)

    def query(self, num_samples: int) -> NDArray[np.bool_]:
        """Select a new set of datapoints to be labeled.

        Args:
            num_samples (int): The number of samples to select.

        Returns:
            NDArray[np.bool_]: A boolean mask for the selected samples.
        """
        unlabeled_features = self.features[~self.labeled_pool]
        probs, vectors = self.model.get_probs_and_embedding(unlabeled_features)
        labels = np.argmax(probs, keepdims=True)
        delta = probs - labels

        if num_samples > len(unlabeled_features):
            raise ValueError(
                f"num_samples ({num_samples}) is greater than unlabeled pool size ({len(unlabeled_features)})"
            )

        centroids = self._select_samples(vectors, delta, num_samples)

        mask = np.zeros(len(self.features), dtype=bool)
        unlabeled_indices = np.flatnonzero(~self.labeled_pool)
        mask[unlabeled_indices[centroids]] = True
        return mask
