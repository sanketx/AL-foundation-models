"""Abstract base class for intial query methods."""


import abc

import numpy as np
from numpy.typing import NDArray


class BaseInit(metaclass=abc.ABCMeta):
    """This class provides the blueprint for different intial query methods."""

    def __init__(
        self, features: NDArray[np.float32], labels: NDArray[np.int64]
    ) -> None:
        """Intialize the class with the feature and label arrays.

        Args:
            features (NDArray[np.float32]): array of input features.
            labels (NDArray[np.int64]): 1D array of target labels.
        """
        self.features = features
        self.labels = labels

    @abc.abstractmethod
    def query(self, num_samples: int) -> NDArray[np.bool_]:
        """Select the intial set of datapoints to be labeled.

        Args:
            num_samples (int): The number of samples to select.

        Returns:
            NDArray[np.bool_]: A boolean mask for the selected samples.
        """
        pass
