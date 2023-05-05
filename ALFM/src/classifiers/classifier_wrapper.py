"""Wrapper class for classifiers to faciliate Active Learning."""

import logging
import warnings
from typing import Dict
from typing import Tuple

import numpy as np
import torch
from ALFM.src.classifiers.registry import ClassifierType
from ALFM.src.datasets.al_dataset import ALDataset
from hydra.utils import instantiate
from numpy import bool_
from numpy.typing import NDArray
from omegaconf import DictConfig


warnings.simplefilter("ignore")
torch.set_float32_matmul_precision("medium")  # type: ignore[no-untyped-call]


class ClassifierWrapper:
    def __init__(self, cfg: DictConfig) -> None:
        self.num_features = cfg.model.num_features
        self.num_classes = cfg.dataset.num_classes

        self.dataloader = instantiate(cfg.dataloader)
        self.trainer_cfg = cfg.trainer

        classifier_type = ClassifierType[cfg.classifier.name]
        classifier_params = instantiate(cfg.classifier.params)

        self.classifier = classifier_type.value(
            self.num_features, num_classes=self.num_classes, **classifier_params
        )

    def fit(
        self,
        features: NDArray[np.float32],
        labels: NDArray[np.int64],
        mask: NDArray[bool_],
    ) -> None:
        total_samples = len(features)
        num_samples = len(features[mask])
        num_classes = len(np.unique(labels))
        seen_classes = len(np.unique(labels[mask]))
        num_features = features.shape[1]

        logging.info(
            f"Training on {num_samples}/{total_samples} samples with dim: "
            + f"{num_features}, seen {seen_classes}/{num_classes} classes"
        )

        dataset = ALDataset(features, labels, mask)
        self.trainer = instantiate(self.trainer_cfg)
        self.trainer.fit(self.classifier, self.dataloader(dataset, shuffle=True))

    def eval(
        self,
        features: NDArray[np.float32],
        labels: NDArray[np.int64],
    ) -> Dict[str, float]:
        dataset = ALDataset(features, labels)
        return self.trainer.test(  # type: ignore[no-any-return]
            self.classifier, self.dataloader(dataset), ckpt_path="best", verbose=False
        )[0]

    def get_probs(
        self, features: NDArray[np.float32], dropout: bool = False
    ) -> NDArray[np.float32]:
        self.classifier.set_pred_mode("probs")
        return self._predict(features, dropout)["probs"]

    def get_embedding(
        self, features: NDArray[np.float32], dropout: bool = False
    ) -> NDArray[np.float32]:
        self.classifier.set_pred_mode("embed")
        return self._predict(features, dropout)["embed"]

    def get_probs_and_embedding(
        self, features: NDArray[np.float32], dropout: bool = False
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        self.classifier.set_pred_mode(["probs", "embed"])
        preds = self._predict(features, dropout)
        return preds["probs"], preds["embed"]

    def _predict(
        self, features: NDArray[np.float32], dropout: bool
    ) -> Dict[str, NDArray[np.float32]]:
        dataset = ALDataset(features, np.zeros(len(features), dtype=np.int64))
        self.classifier.set_dropout(dropout)

        preds = self.trainer.predict(
            self.classifier, self.dataloader(dataset), ckpt_path="best"
        )
        return {key: np.concatenate([p[key] for p in preds]) for key in preds[0]}
