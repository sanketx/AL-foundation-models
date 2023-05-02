"""Base class for classifiers."""


from typing import Dict
from typing import List
from typing import Literal
from typing import Tuple
from typing import cast

import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.optim import Optimizer
from torchmetrics import Metric


class BaseClassifier(LightningModule):
    """This class provides the blueprint for different classifiers."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        dropout_p: float,
        lr: float,
        weight_decay: float,
        metrics: List[Metric],
    ) -> None:
        """Intialize the model parameters."""
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_p)  # for MC sampling
        self.feature_extractor: nn.Module  # define this in subclasses
        self.linear = nn.Linear(input_dim, num_classes)

        self.lr = lr
        self.weight_decay = weight_decay

        self.loss = nn.CrossEntropyLoss()
        self.metrics = nn.ModuleDict(
            {
                stage: nn.ModuleList([metric.clone() for metric in metrics])
                for stage in ["TRAIN", "VAL", "TEST"]
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = self.linear(x)
        return x

    def step(
        self, batch: torch.Tensor, stage: Literal["TRAIN", "VAL", "TEST"]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = batch
        y = y.squeeze()

        y_pred = self(x)  # logits
        y_prob = y_pred.softmax(dim=-1)  # class probabilities

        for metric in self.metrics[stage]:
            metric(y_prob, y)
            self.log(
                f"{stage}_{type(metric).__name__}",
                metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        return y_pred, y

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        y_pred, y = self.step(batch, "TRAIN")

        loss = self.loss(y_pred, y)
        self.log("CELoss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return cast(torch.Tensor, loss)

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        self.step(batch, "VAL")

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        self.step(batch, "TEST")

    def configure_optimizers(self) -> Optimizer:
        return torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def set_pred_mode(
        self, mode: Literal["probs", "embed"] | List[Literal["probs", "embed"]]
    ) -> None:
        self.pred_mode = mode if isinstance(mode, list) else [mode]

    def predict_step(  # type: ignore[override]
        self, batch: torch.Tensor, batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        x, _ = batch
        embedding = self.feature_extractor(x)
        probs = self.linear(embedding).softmax(dim=1)

        tensors = {"embed": embedding, "probs": probs}
        return {k: tensors[k].float().cpu().numpy() for k in self.pred_mode}
