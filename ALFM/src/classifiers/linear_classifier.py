"""Linear classification model."""

from typing import Any

from ALFM.src.classifiers.base_classifier import BaseClassifier
from torch import nn


class LinearClassifier(BaseClassifier):
    def __init__(self, input_dim: int, **params: Any) -> None:
        super().__init__(input_dim, **params)
        self.feature_extractor = nn.Sequential(
            self.dropout, nn.BatchNorm1d(input_dim, affine=False)
        )
