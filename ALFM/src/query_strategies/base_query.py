"""Abstract base class for Active Learning query methods."""

import abc

import numpy as np
from numpy.typing import NDArray

from ALFM.src.classifiers.classifier_wrapper import ClassifierWrapper


class BaseQuery(metaclass=abc.ABCMeta):
    """This class provides the blueprint for different intial query methods."""

    def __init__(
        self, features: NDArray[np.float32], labels: NDArray[np.int64]
    ) -> None:
        """Intialize the class at iteration 1."""
        self.iteration = 1
        self.features = features
        self.labels = labels

    def update_state(
        self,
        iteration: int,
        labeled_pool: NDArray[np.bool_],
        model: ClassifierWrapper,
    ) -> None:
        """Update the experiment state for the query function.

        Args:
            iteration (int): The current AL iteration
            labeled_pool (NDArray[np.bool_]): The mask representing the labeled pool.
            model (ClassifierWrapper): The updated current model.
        """
        self.iteration = iteration
        self.labeled_pool = labeled_pool
        self.model = model

    @abc.abstractmethod
    def query(self, num_samples: int) -> NDArray[np.bool_]:
        """Select a new set of datapoints to be labeled.

        Args:
            num_samples (int): The number of samples to select.

        Returns:
            NDArray[np.bool_]: A boolean mask for the selected samples.
        """
        pass
