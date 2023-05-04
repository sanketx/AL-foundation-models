"""Experiment logger for Active Learning experiments."""

import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any
from typing import Dict

import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig
from omegaconf import OmegaConf
from rich.pretty import pretty_repr


class ExperimentLogger:
    """Experiment Logger class to log experiments."""

    def __init__(self, log_dir: str, cfg: DictConfig) -> None:
        self.exp_dir = Path(log_dir) / "configs" / cfg.dataset.name / cfg.model.name
        self.csv_dir = Path(log_dir) / "results" / cfg.dataset.name / cfg.model.name

        if not os.path.exists(log_dir):
            logging.info(f"Creating log dir {log_dir}")
            os.makedirs(log_dir)

        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.csv_dir, exist_ok=True)

        logging.info(f"Saving logs to {log_dir}")
        self.log_cfg(OmegaConf.to_container(cfg, resolve=True))  # type: ignore[arg-type]

    def log_cfg(self, cfg: Dict[str, Any]) -> None:
        del cfg["trainer"]
        del cfg["dataloader"]
        del cfg["classifier"]["params"]["metrics"]

        force_exp = cfg.pop("force_exp")
        logging.info(f"Experiment Parameters: {pretty_repr(cfg)}")

        json_str = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
        self.hash_str = hashlib.md5(json_str.encode("utf-8")).hexdigest()

        file_name = f"{cfg['query_strategy']['name']}-{self.hash_str}.yaml"
        exp_file = self.exp_dir / file_name

        if os.path.exists(exp_file) and not force_exp:
            logging.error(
                f"A config file with these parameters exists: '{file_name}'."
                + "\nSpecify 'force_exp=true' to override"
            )
            sys.exit(1)

        if os.path.exists(exp_file) and force_exp:
            logging.warning(
                f"A config file with these parameters exists: '{file_name}'."
                + "\nOverwriting previous experiment's results"
            )

        logging.info(f"Saving parameters to '{file_name}'")
        OmegaConf.save(cfg, exp_file)

    def log_composition(
        self,
        features: NDArray[np.float32],
        labels: NDArray[np.int64],
        mask: NDArray[np.bool_],
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

    def log_scores(
        self, scores: Dict[str, float], i: int, num_iter: int, budget: int
    ) -> None:
        logging.info(
            f"[{i}/{num_iter}] Budget: {budget} "
            + f"| Acc: {scores['TEST_MulticlassAccuracy']:.4f}"
            + f" | AUROC: {scores['TEST_MulticlassAUROC']:.4f}"
        )
