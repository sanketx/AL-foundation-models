"""Coreset query strategy."""


import numpy as np
from ALFM.src.query_strategies.base_query import BaseQuery
from numpy.typing import NDArray
from faiss import pairwise_distances


class Coreset(BaseQuery):
    """Selects diverse samples that cover the unlabeled feature space
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

    def update_distances(self, cluster_centers, only_new=True, reset_dist=False):
        """Update min distances given cluster centers.
        
        Args:
            cluster_centers: indices of cluster centers
            only_new: only calculate distance for newly selected points and
                      update min_distances.
            rest_dist: whether to reset min_distances.
        """
        if reset_dist:
            self.min_distances = None
        if only_new:
            cluster_centers = [d for d in cluster_centers
                               if d not in np.flatnonzero(self.labeled_pool)]
        if cluster_centers:
            x = self.features[cluster_centers]
            dist = pairwise_distances(self.features, x)

        if self.min_distances is None:
            self.min_distances = np.min(dist, axis=1).reshape(-1, 1)
        else:
            self.min_distances = np.minimum(self.min_distances, dist)

    def query(self, num_samples: int) -> NDArray[np.bool_]:
        """Select a new set of datapoints to be labeled.

        Args:
            num_samples (int): The number of samples to select.

        Returns:
            NDArray[np.bool_]: A boolean mask for the selected samples.
        """
        mask = np.zeros(len(self.features), dtype=bool)
        labeled_indices = np.flatnonzero(self.labeled_pool)
        unlabeled_indices = np.flatnonzero(~self.labeled_pool)

        if num_samples > len(unlabeled_indices):
            raise ValueError(
                f"num_samples ({num_samples}) is greater than unlabeled pool size ({len(unlabeled_indices)})"
            )

        try:
            self.update_distances(labeled_indices, only_new=False, reset_dist=True)
        except Exception:
            self.update_distances(labeled_indices, only_new=True, reset_dist=False)

        new_batch = []
        for _ in range(num_samples):
            if self.already_selected is None:
                # Initialize centers with a randomly selected datapoint
                ind = np.random.choice(np.arange(self.n_obs))
            else:
                ind = np.argmax(self.min_distances)

            # New examples should not be in already selected since those points
            # should have min_distance of zero to a cluster center.
            assert ind not in labeled_indices

            self.update_distances([ind], only_new=True, reset_dist=False)
            new_batch.append(ind)

        print('Maximum distance from cluster centers is %0.2f'
              % max(self.min_distances))

        mask[new_batch] = True
        return mask
