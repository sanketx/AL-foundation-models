"""Base class for classifiers."""


from typing import List

import torch
from pytorch_lightning import LightningModule
from torch import nn
from torchmetrics import Metric


class BaseClassifier(LightningModule):
    """This class provides the blueprint for different classifiers."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        dropout_p: float,
        stem: nn.Module,
        lr: float,
        weight_decay: float,
        metrics: List[Metric],
    ) -> None:
        """Intialize the model parameters."""
        self.dropout = nn.Dropout(p=dropout_p)
        self.stem = stem
        self.norm = nn.BatchNorm1d(input_dim, affine=False)
        self.linear = nn.Linear(input_dim, num_classes)
        self.loss = nn.CrossEntropyLoss()

        self.lr = lr
        self.weight_decay = weight_decay

        self.train_metrics = nn.ModuleList([metric.clone() for metric in metrics])
        self.test_metrics = nn.ModuleList([metric.clone() for metric in metrics])

    def forward(self, x: torch.Tensor):
        x = self.dropout(x)
        x = self.stem(x)
        x = self.norm(x)
        x = self.linear(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_pred = self(x).softmax(dim=-1)
        loss = self.loss(y_pred, y)

        # Update train metrics
        for metric in self.train_metrics:
            metric(y_pred, y)

        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
