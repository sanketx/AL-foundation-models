"""Centroid init query class."""

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

from ALFM.src.clustering.kmeans import cluster_features
from ALFM.src.init_strategies.base_init import BaseInit


class CentroidInit(BaseInit):
    """Select samples based on their typicality scores."""

    def __init__(self, **params: Any) -> None:
        """Intialize the class with the feature and label arrays.

        Args:
            features (NDArray[np.float32]): array of input features.
            labels (NDArray[np.int64]): 1D array of target labels.
        """
        super().__init__(**params)

    def query(self, num_samples: int) -> NDArray[np.bool_]:
        """Select the intial set of datapoints to be labeled.

        Args:
            num_samples (int): The number of samples to select.

        Returns:
            NDArray[np.bool_]: A boolean mask for the selected samples.
        """
        if num_samples > len(self.features):
            raise ValueError(
                f"num_samples ({num_samples}) is greater than dataset size ({len(self.features)})"
            )

        features = torch.from_numpy(self.features)
        vectors = F.normalize(features).numpy()
        selected = cluster_features(vectors, int(num_samples))

        mask = np.zeros(len(self.features), dtype=bool)
        mask[selected] = True
        return mask
