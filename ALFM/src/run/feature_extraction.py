"""Feature extraction and caching using pretraing backbones."""

import logging
import os
from multiprocessing.shared_memory import SharedMemory
from typing import Any
from typing import Callable
from typing import Optional
from typing import Sequence
from typing import Tuple

import h5py
import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from numpy.typing import NDArray
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from ALFM.src.datasets.factory import create_dataset
from ALFM.src.datasets.registry import DatasetType
from ALFM.src.models.backbone_wrapper import BackboneWrapper
from ALFM.src.models.factory import create_model
from ALFM.src.models.registry import ModelType


logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
torch.set_float32_matmul_precision("medium")  # type: ignore[no-untyped-call]


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

        self.feature_shm = SharedMemory(
            create=True, size=4 * num_samples * num_features
        )
        self.label_shm = SharedMemory(create=True, size=8 * num_samples)

        self.features: NDArray[np.float32] = np.ndarray(
            (num_samples, num_features),
            dtype=np.float32,
            buffer=self.feature_shm.buf,
        )
        self.labels: NDArray[np.int64] = np.ndarray(
            (num_samples, 1),
            dtype=np.int64,
            buffer=self.label_shm.buf,
        )

        self.features[:] = -1
        self.labels[:] = -1

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
        self.features[batch_indices] = predictions[0]
        self.labels[batch_indices] = predictions[1]

    def get_predictions(self) -> Tuple[NDArray[np.float32], NDArray[np.int64]]:
        """Return detached copies of prediction vectors."""
        return self.features, self.labels

    def close(self) -> None:
        """Release shared memory."""
        self.feature_shm.close()
        self.feature_shm.unlink()
        self.label_shm.close()
        self.label_shm.unlink()


def check_existing_features(vector_file: str, split: str) -> bool:
    """Check if the features for the specified split already exist.

    Args:
        vector_file (str): Path to the HDF file containing the features.
        split (str): Split name, either 'train' or 'test'.

    Raises:
        RuntimeError: If the features for the specified split already exist in the HDF file.
    """
    if os.path.exists(vector_file):
        with h5py.File(vector_file, "r") as fh:
            return split in fh.keys()

    return False


def save_vectors(
    features: NDArray[np.float32],
    labels: NDArray[np.int64],
    vector_file: str,
    split: str,
) -> None:
    """Save the extracted features to an HDF file.

    Args:
        features (NDArray[np.float32]): Extracted features as a NumPy array.
        labels (NDArray[np.int64]): Image labels as a NumPy array.
        vector_file (str): Path to the HDF file to save the features.
        split (str): Split name, either 'train' or 'test'.
    """
    with h5py.File(vector_file, "a") as fh:
        fh.create_dataset(f"{split}/features", data=features)
        fh.create_dataset(f"{split}/labels", data=labels)


def extract_features(
    dataset_cfg: DictConfig,
    train: bool,
    model_cfg: DictConfig,
    dataset_dir: str,
    model_dir: str,
    feature_dir: str,
    dataloader: Callable[..., DataLoader[Any]],
    trainer_cfg: DictConfig,
) -> None:
    """Extract features from the dataset using the specified pretrained model and save them to disk.

    Args:
        dataset_cfg (DictConfig): Config representing the dataset to use.
        train (bool): True if extracting features for the training set, False for the test set.
        model_cfg (DictConfig): Config representing the pretrained model to use.
        dataset_dir (str): Path to the directory containing the dataset.
        model_dir (str): Path to the directory containing the model cache.
        feature_dir (str): Path to the directory where extracted features will be saved.
        dataloader (Callable[..., DataLoader[Any]]): Callable to create DataLoader for the dataset.
        trainer_cfg (pl.Trainer): PyTorch Lightning Trainer config for feature extraction.
    """
    dataset_type = DatasetType[dataset_cfg.name]
    model_type = ModelType[model_cfg.name]

    split = "train" if train else "test"
    dst_dir = os.path.join(feature_dir, f"{dataset_type.name}")
    vector_file = os.path.join(dst_dir, f"{model_type.name}.hdf")
    os.makedirs(dst_dir, exist_ok=True)

    if check_existing_features(vector_file, split):
        logging.warn(
            f"{split} features have already been computed for the {dataset_type.name}"
            + f" dataset with the {model_type.name} model"
        )
        return  # skip feature extraction

    model, transform = create_model(model_type, cache_dir=model_dir)
    dataset = create_dataset(
        dataset_type, root=dataset_dir, train=train, transform=transform
    )

    model = BackboneWrapper(model)
    shm_writer = SharedMemoryWriter(
        len(dataset), dataset_cfg.num_classes, model_cfg.num_features
    )

    prog_bar = pl.callbacks.RichProgressBar()
    trainer = hydra.utils.instantiate(trainer_cfg, callbacks=[shm_writer, prog_bar])
    trainer.predict(model, dataloader(dataset), return_predictions=False)

    features, labels = shm_writer.get_predictions()
    save_vectors(features, labels, vector_file, split)
    shm_writer.close()
