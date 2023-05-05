"""Experiment logger for Active Learning experiments."""

import csv
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
        hash_str = hashlib.blake2b(json_str.encode("utf-8"), digest_size=8).hexdigest()

        self.file_name = f"{cfg['query_strategy']['name']}-{hash_str}"
        exp_file = self.exp_dir / f"{self.file_name}.yaml"

        if os.path.exists(exp_file) and not force_exp:
            logging.error(
                f"A config file with these parameters exists: '{self.file_name}.yaml'."
                + "\nSpecify 'force_exp=true' to override"
            )
            sys.exit(1)

        if os.path.exists(exp_file) and force_exp:
            logging.warning(
                f"A config file with these parameters exists: '{self.file_name}.yaml'."
                + "\nOverwriting previous experiment's results"
            )

            csv_file = self.csv_dir / f"{self.file_name}.csv"

            if os.path.isfile(csv_file):
                os.remove(csv_file)  # remove previous experiment's results

        logging.info(f"Saving parameters to '{self.file_name}.yaml'")
        OmegaConf.save(cfg, exp_file)

    def log_scores(
        self, scores: Dict[str, float], iteration: int, num_iter: int, num_samples: int
    ) -> None:
        logging.info(
            f"[{iteration}/{num_iter}] Training samples: {num_samples} "
            + f"| Acc: {scores['TEST_MulticlassAccuracy']:.4f}"
            + f" | AUROC: {scores['TEST_MulticlassAUROC']:.4f}"
        )

        fields = ["iteration", "num_samples"] + list(scores.keys())
        data = {"iteration": iteration, "num_samples": num_samples} | scores
        csv_file = self.csv_dir / f"{self.file_name}.csv"
        file_exists = os.path.isfile(csv_file)

        with open(csv_file, mode="a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)

            if not file_exists:
                writer.writeheader()

            writer.writerow(data)
            fh.flush()
