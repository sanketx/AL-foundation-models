"""Coreset query strategy."""


import numpy as np
from ALFM.src.query_strategies.base_query import BaseQuery
from faiss import pairwise_distances
from numpy.typing import NDArray
from rich.progress import track


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

    def __init__(self) -> None:
        """Call the superclass constructor."""
        super().__init__()

    def query(self, num_samples: int) -> NDArray[np.bool_]:
        """Select a new set of datapoints to be labeled.

        Args:
            num_samples (int): The number of samples to select.

        Returns:
            NDArray[np.bool_]: A boolean mask for the selected samples.
        """
        mask = np.zeros(len(self.features), dtype=bool)
        labeled_features = self.features[self.labeled_pool]
        unlabeled_features = self.features[~self.labeled_pool]

        if num_samples > len(unlabeled_features):
            raise ValueError(
                f"num_samples ({num_samples}) is greater than unlabeled pool size ({len(unlabeled_features)})"
            )

        p_dist = pairwise_distances(labeled_features, unlabeled_features)
        min_dist = p_dist.min(axis=0)  # distance of each UL point to the nearest center

        new_batch = []

        for _ in track(range(num_samples), description="[green]Core-Set query"):
            next_center = np.argmax(min_dist)
            new_batch.append(next_center)

            new_dist = pairwise_distances(
                unlabeled_features[next_center].reshape(1, -1),
                unlabeled_features,
            )
            min_dist = np.minimum(min_dist, new_dist.ravel())

        unlabeled_indices = np.flatnonzero(~self.labeled_pool)
        mask[unlabeled_indices[new_batch]] = True
        return mask
