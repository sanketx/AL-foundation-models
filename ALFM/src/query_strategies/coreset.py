"""Coreset query strategy."""

from typing import Any

import numpy as np
import torch
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
        labeled_vectors = vectors[self.labeled_pool].cuda().unsqueeze(0)
        unlabeled_vectors = vectors[~self.labeled_pool].cuda().unsqueeze(0)

        if num_samples > unlabeled_vectors.shape[1]:
            raise ValueError(
                f"num_samples ({num_samples}) is greater than unlabeled pool size ({unlabeled_vectors.shape[1]})"
            )

        # pairwise_distances in FAISS returns the squared L2 distance
        p_dist = torch.cdist(labeled_vectors, unlabeled_vectors).squeeze()
        min_dist = p_dist.min(dim=0)[0]  # distance of UL points to the nearest center

        new_batch = []

        for _ in track(range(num_samples), description="[green]Core-Set query"):
            next_center = min_dist.argmax()
            new_batch.append(next_center.item())

            new_dist = torch.cdist(
                unlabeled_vectors[0, next_center].view(1, 1, -1),
                unlabeled_vectors,
            )
            min_dist = torch.minimum(min_dist, new_dist.ravel())

        mask = np.zeros(len(vectors), dtype=bool)
        unlabeled_indices = np.flatnonzero(~self.labeled_pool)
        mask[unlabeled_indices[new_batch]] = True
        return mask
