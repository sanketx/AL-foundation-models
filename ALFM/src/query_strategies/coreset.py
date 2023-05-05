"""Coreset query strategy."""

from typing import Any

import numpy as np
from faiss import pairwise_distances
from numpy.typing import NDArray
from rich.progress import track

from ALFM.src.query_strategies.base_query import BaseQuery


class Coreset(BaseQuery):
    """Coreset Active Learning query function.

    Selects diverse samples that cover the unlabeled feature space
    according to the coreset algorithm as described in Sener and Savarese,
    "Active Learning for Convolutional Neural Networks: A Core-Set Approach"
    (https://arxiv.org/abs/1708.00489).

    We implement the greedy approach, which performs comparably to the more
    involved mixed-integer programming approach. Adapted from
    https://github.com/google/active-learning/blob/master/sampling_methods/kcenter_greedy.py.
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
        vectors = self.model.get_embedding(self.features)
        labeled_vectors = vectors[self.labeled_pool]
        unlabeled_vectors = vectors[~self.labeled_pool]

        if num_samples > len(unlabeled_vectors):
            raise ValueError(
                f"num_samples ({num_samples}) is greater than unlabeled pool size ({len(unlabeled_vectors)})"
            )

        # pairwise_distances in FAISS returns the squared L2 distance
        p_dist = pairwise_distances(labeled_vectors, unlabeled_vectors)
        min_dist = p_dist.min(axis=0)  # distance of each UL point to the nearest center

        new_batch = []

        for _ in track(range(num_samples), description="[green]Core-Set query"):
            next_center = np.argmax(min_dist)
            new_batch.append(next_center)

            new_dist = pairwise_distances(
                unlabeled_vectors[next_center].reshape(1, -1),
                unlabeled_vectors,
            )
            min_dist = np.minimum(min_dist, new_dist.ravel())

        mask = np.zeros(len(vectors), dtype=bool)
        unlabeled_indices = np.flatnonzero(~self.labeled_pool)
        mask[unlabeled_indices[new_batch]] = True
        return mask
