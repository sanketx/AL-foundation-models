"""Residual Adapter model."""

from typing import List

import torch
import torch.nn.functional as F
from ALFM.src.classifiers.base_classifier import BaseClassifier
from torch import Tensor
from torch import nn
from torchmetrics import Metric


class ResidualAdapter(BaseClassifier):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        dropout_p: float,
        lr: float,
        weight_decay: float,
        metrics: List[Metric],
        scale: int,
        init_alpha: float,
    ) -> None:
        super().__init__(input_dim, num_classes, dropout_p, lr, weight_decay, metrics)
        self.batch_norm = nn.BatchNorm1d(input_dim, affine=False)
        self.alpha = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32))

        self.stem = nn.Sequential(  # bottleneck MLP
            nn.Linear(input_dim, input_dim // scale),
            nn.GELU(),
            nn.Linear(input_dim // scale, input_dim),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.dropout(x)

        x_adapt = self.stem(x)
        alpha = torch.sigmoid(self.alpha)
        x = alpha * x_adapt + (1 - alpha) * x

        x = self.batch_norm(x)
        x = self.linear(x)
        return x
