"""Wrapper module for a PyTorch model for use with PyTorch Lightning."""

from typing import cast

import numpy as np
import numpy.typing as npt
import pytorch_lightning as pl
import torch
from torch import nn


class BackboneWrapper(pl.LightningModule):
    """PyTorch Lightning module wrapper for a PyTorch model.

    Args:
        model (nn.Module): PyTorch model to be wrapped.
    """

    def __init__(self, model: nn.Module) -> None:
        """Initialize the BackboneWrapper.

        Args:
            model (nn.Module): PyTorch model to be wrapped.
        """
        super().__init__()
        self.model = model

    def predict_step(
        self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = 0
    ) -> npt.NDArray[np.float32 | np.float16]:
        """Prediction step for a batch of data.

        Args:
            batch (torch.Tensor): Input batch of data.
            batch_idx (int): Index of the current batch.
            dataloader_idx (int, optional): Index of the current dataloader.

        Returns:
            npt.NDArray[np.float32 | np.float16]: Model predictions for the input batch.
        """
        x, y = batch
        predictions = self.model(x).cpu().numpy()
        return cast(npt.NDArray[np.float32 | np.float16], predictions)