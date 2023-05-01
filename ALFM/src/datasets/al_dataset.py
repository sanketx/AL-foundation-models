"""Vector Dataset class used in ALFM experiments."""

from typing import Any
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from torch.utils.data import Dataset


class ALDataset(Dataset[Any]):
    """Vector Dataset class used in ALFM experiments.

    The dataset is created from a set of features, labels, and a mask indicating
    which instances are to be used. The mask is used to select a subset of the data
    to be included in the dataset.
    """

    def __init__(
        self,
        features: NDArray[np.float32],
        labels: NDArray[np.int64],
        mask: NDArray[np.bool_],
    ) -> None:
        """Creates a dataset instance from the provided features, labels, and mask.

        Args:
            features (NDArray[np.float32]): A NumPy array of feature vectors.
            labels (NDArray[np.int64]): A NumPy array of corresponding labels.
            mask (NDArray[np.bool_]): A NumPy boolean array used to filter the data points .
        """
        self.features = features
        self.labels = labels
        self.indices = np.flatnonzero(mask)

    def __len__(self) -> int:
        """Returns the length of the dataset after applying the mask.

        Returns:
            int: The number of data points in the masked dataset.
        """
        return len(self.indices)

    def __getitem__(self, index: int) -> Tuple[NDArray[np.float32], NDArray[np.int64]]:
        """Retrieves a single data point from the dataset based on the provided index.

        Args:
            index (int): The index of the desired data point.

        Returns:
            Tuple[NDArray[np.float32], NDArray[np.int64]]: A tuple containing the
                feature vector and the corresponding label for the requested data point.
        """
        true_index = self.indices[index]
        return self.features[true_index], self.labels[true_index]
