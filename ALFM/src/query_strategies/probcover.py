"""Implements ProbCover query strategy."""

from typing import Any

import numpy as np
from faiss import pairwise_distances
from numpy.typing import NDArray

from ALFM.src.query_strategies.base_query import BaseQuery


class ProbCover(BaseQuery):
    """ProbCover active learning query function.

    Selects samples that cover the unlabeled feature space according to the
    ProbCover algorithm as described in Yehuda et al. "Active Learning
    Through a Covering Lens" (https://arxiv.org/pdf/2205.11320.pdf).
    """

    def __init__(self, delta: int, **params: Any) -> None:
        """Call the superclass constructor."""
        super().__init__(**params)
        self.delta = delta
        self.delta_graph = self.construct_graph()

    def construct_graph(self) -> None:
        """Adapted from https://github.com/avihu111/TypiClust/blob/main/deep-al/pycls/al/prob_cover.py

        Creates a graph containing edges between features where
        l2(feature1, feature2) < delta.
        """
        vectors = self.model.get_embedding(self.features)
        dist = pairwise_distances(vectors, vectors)
        dist = np.sqrt(dist)
        np.fill_diagonal(dist, np.inf)  # remove self distances
        mask = dist < self.delta  # get distances less than delta
        xs, ys = np.where(mask)
        ds = dist[xs, ys]

        print(f"Finished constructing graph using delta={self.delta}")
        return {"x": xs, "y": ys, "d": ds}

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

        indices = []
        edge_from_seen = np.isin(self.delta_graph["x"], labeled_indices)
        covered_samples = np.unique(self.delta_graph["y"][edge_from_seen])
        # remove incoming edges to all covered samples from the existing labeled set
        cur = {
            "x": self.delta_graph["x"][
                ~np.isin(self.delta_graph["y"], covered_samples)
            ],
            "y": self.delta_graph["y"][
                ~np.isin(self.delta_graph["y"], covered_samples)
            ],
            "d": self.delta_graph["d"][
                ~np.isin(self.delta_graph["y"], covered_samples)
            ],
        }

        for i in range(num_samples):
            coverage = len(covered_samples) / (len(self.features))
            degrees = np.bincount(cur["x"], minlength=len(self.features))
            print(
                f'Coverage: {coverage:.2f}, Max degree: {np.max(degrees):.2f}, Edge count: {len(cur["x"])}'
            )

            sample = np.argmax(degrees)
            # sample = np.random.choice(np.argsort(degrees)[::-1][:5])  # randomize selection from top 5
            new_covered_samples = cur["y"][cur["x"] == sample]

            assert (
                len(np.intersect1d(new_covered_samples, covered_samples)) == 0
            ), "New covered samples should not be in the covered set."

            cur = {
                "x": cur["x"][~np.isin(cur["y"], covered_samples)],
                "y": cur["y"][~np.isin(cur["y"], covered_samples)],
                "d": cur["d"][~np.isin(cur["y"], covered_samples)],
            }
            covered_samples = np.concatenate([covered_samples, new_covered_samples])
            indices.append(sample)

        assert (
            len(indices) == num_samples
        ), "Number of selected samples does not match num_samples."
        mask[indices] = True
        return mask
