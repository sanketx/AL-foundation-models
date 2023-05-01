"""Wrapper class for classifiers to faciliate Active Learning."""
import warnings
from typing import Dict

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
            self.num_features, self.num_classes, **classifier_params
        )

    def fit(
        self,
        train_x: NDArray[np.float32],
        train_y: NDArray[np.int64],
        labeled_pool: NDArray[bool_],
    ) -> None:
        dataset = ALDataset(train_x, train_y, labeled_pool)
        self.trainer = instantiate(self.trainer_cfg)
        self.trainer.fit(self.classifier, self.dataloader(dataset, shuffle=True))

    def eval(
        self,
        test_x: NDArray[np.float32],
        test_y: NDArray[np.int64],
    ) -> Dict[str, float]:
        dataset = ALDataset(test_x, test_y)
        return self.trainer.test(  # type: ignore[no-any-return]
            self.classifier, self.dataloader(dataset), ckpt_path="best", verbose=False
        )[0]
