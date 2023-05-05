"""Random init query class."""

import numpy as np
from numpy.typing import NDArray

from ALFM.src.init_strategies.base_init import BaseInit


class RandomInit(BaseInit):
    """Randomly select an inital pool of samples to be labeled."""

    def __init__(
        self, features: NDArray[np.float32], labels: NDArray[np.int64]
    ) -> None:
        """Intialize the class with the feature and label arrays.

        Args:
            features (NDArray[np.float32]): array of input features.
            labels (NDArray[np.int64]): 1D array of target labels.
        """
        super().__init__(features, labels)

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

        mask = np.zeros(len(self.features), dtype=bool)
        indices = np.random.choice(len(self.features), size=num_samples, replace=False)
        mask[indices] = True
        return mask
