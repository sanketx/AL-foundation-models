"""Residual Adapter model."""

from typing import Any

import torch
from ALFM.src.classifiers.base_classifier import BaseClassifier
from torch import nn


class ResidualMLP(nn.Module):
    def __init__(self, input_dim: int, scale: int, init_alpha: float) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32))

        self.mlp = nn.Sequential(  # bottleneck MLP
            nn.Linear(input_dim, input_dim // scale),
            nn.GELU(),
            nn.Linear(input_dim // scale, input_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_adapt = self.mlp(x)
        alpha = torch.sigmoid(self.alpha)
        x = alpha * x_adapt + (1 - alpha) * x
        return x


class ResidualAdapter(BaseClassifier):
    def __init__(
        self, input_dim: int, scale: int, init_alpha: float, **params: Any
    ) -> None:
        super().__init__(input_dim, **params)
        self.feature_extractor = nn.Sequential(
            ResidualMLP(input_dim, scale, init_alpha),
            nn.BatchNorm1d(input_dim, affine=False),
        )
