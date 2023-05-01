"""Linear classification model."""

from typing import List

from ALFM.src.classifiers.base_classifier import BaseClassifier
from torch import Tensor
from torch import nn
from torchmetrics import Metric


class LinearClassifier(BaseClassifier):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        dropout_p: float,
        lr: float,
        weight_decay: float,
        metrics: List[Metric],
    ) -> None:
        super().__init__(input_dim, num_classes, dropout_p, lr, weight_decay, metrics)
        self.batch_norm = nn.BatchNorm1d(input_dim, affine=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.dropout(x)
        x = self.batch_norm(x)
        x = self.linear(x)
        return x
