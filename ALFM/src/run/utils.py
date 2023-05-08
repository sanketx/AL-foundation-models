"""Experiment logger for Active Learning experiments."""

import csv
import hashlib
import json
import logging
import os
import sys
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy as np
import pytorch_lightning as pl
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


class SharedMemoryWriter(pl.callbacks.BasePredictionWriter):
    """Writes multi-GPU predictions to shared memory."""

    def __init__(self, num_samples: int, num_classes: int, num_features: int) -> None:
        """Create a new SharedMemoryWriter callback.

        Args:
            num_samples (int): number of samples in the dataset.
            num_classes (int): number of classes in the dataset.
            num_features (int): number of features in the dataset.
        """
        super().__init__(write_interval="batch")
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.num_features = num_features

        self.feature_shm, self.label_shm = self._get_shm()
        self.features, self.labels = self._get_arrays()

    def write_on_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        predictions: Any,
        batch_indices: Optional[Sequence[Any]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Write predictions from each process to shared memory."""
        self.local_rank = trainer.local_rank  # for SHM cleanup later
        self.features[batch_indices] = predictions[0]
        self.labels[batch_indices] = predictions[1]

    def get_predictions(self) -> Tuple[NDArray[np.float32], NDArray[np.int64]]:
        """Return prediction vectors."""
        return self.features, self.labels

    def close(self) -> None:
        """Release shared memory.

        Only call this from the rank 0 process as multiple calls to close will
        raise an exception.
        """
        if self.local_rank == 0:
            self.feature_shm.close()
            self.feature_shm.unlink()
            self.label_shm.close()
            self.label_shm.unlink()

    def _get_names(self) -> Tuple[str, str]:
        """Get a unique name for shared memory blocks.

        Distributed Data Parallel creates copies of the parent process. We want all
        instances of the SharedMemoryWriter to write to the same block of shared
        memory. The parent and child processes all share the same process group ID.
        This ID is used to create a common name for all the parallel processes.
        """
        pgid = os.getpgid(0)
        return f"feature-{pgid}", f"label-{pgid}"

    def _get_shm(self) -> Tuple[SharedMemory, SharedMemory]:
        """Get the shared memory blocks for the SharedMemoryWriter.

        The first process to enter this section will attempt to allocate the block
        of memory. If a process fails to create a block because it already exists,
        it will simply return a handle to that block.
        """
        feature_name, label_name = self._get_names()

        try:
            feature_shm = SharedMemory(
                create=True,
                size=4 * self.num_samples * self.num_features,
                name=feature_name,
            )
        except FileExistsError:
            feature_shm = SharedMemory(feature_name)

        try:
            label_shm = SharedMemory(
                create=True, size=8 * self.num_samples, name=label_name
            )
        except FileExistsError:
            label_shm = SharedMemory(label_name)

        return feature_shm, label_shm

    def _get_arrays(self) -> Tuple[NDArray[np.float32], NDArray[np.int64]]:
        """Create NumPy arrays backed by a shared memory block."""
        labels: NDArray[np.int64] = np.ndarray(
            (self.num_samples, 1), dtype=np.int64, buffer=self.label_shm.buf
        )
        features: NDArray[np.float32] = np.ndarray(
            (self.num_samples, self.num_features),
            dtype=np.float32,
            buffer=self.feature_shm.buf,
        )

        return features, labels
