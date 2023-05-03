"""BADGE sampling class."""

import gc
from typing import Any

import numpy as np
from ALFM.src.clustering.kmeans import kmeans_plus_plus_init
from ALFM.src.query_strategies.base_query import BaseQuery
from numpy.typing import NDArray


class BADGE(BaseQuery):
    """BADGE Active Learning - https://arxiv.org/abs/1906.03671."""

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
        unlabeled_features = self.features[~self.labeled_pool]
        grad_vectors = self.model.get_grad_embedding(unlabeled_features)

        if num_samples > len(unlabeled_features):
            raise ValueError(
                f"num_samples ({num_samples}) is greater than unlabeled pool size ({len(unlabeled_features)})"
            )

        centroids = kmeans_plus_plus_init(grad_vectors, num_samples)
        del grad_vectors
        gc.collect()

        mask = np.zeros(len(self.features), dtype=bool)
        unlabeled_indices = np.flatnonzero(~self.labeled_pool)
        mask[unlabeled_indices[centroids]] = True
        return mask
