"""Feature extraction and caching using pretraing backbones."""

import os
from typing import Any
from typing import Callable

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
from ALFM import DatasetType
from ALFM import ModelType
from ALFM.src.datasets.factory import create_dataset
from ALFM.src.models.backbone_wrapper import BackboneWrapper
from ALFM.src.models.factory import create_model
from numpy.typing import NDArray
from torch.utils.data import DataLoader


torch.set_float32_matmul_precision("medium")  # type: ignore[no-untyped-call]


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
    features: NDArray[np.float32 | np.float16],
    labels: NDArray[np.int64],
    vector_file: str,
    split: str,
) -> None:
    """Save the extracted features to an HDF file.

    Args:
        features (NDArray[np.float32 | np.float16]): Extracted features as a NumPy array.
        labels (NDArray[np.int64]): Image labels as a NumPy array.
        vector_file (str): Path to the HDF file to save the features.
        split (str): Split name, either 'train' or 'test'.
    """
    with h5py.File(vector_file, "a") as fh:
        fh.create_dataset(f"{split}/features", data=features)
        fh.create_dataset(f"{split}/labels", data=labels)


def extract_features(
    dataset_type: DatasetType,
    train: bool,
    model_type: ModelType,
    dataset_dir: str,
    model_dir: str,
    feature_dir: str,
    dataloader: Callable[..., DataLoader[Any]],
    trainer: pl.Trainer,
) -> None:
    """Extract features from the dataset using the specified pretrained model and save them to disk.

    Args:
        dataset_type (DatasetType): Enum representing the type of dataset to use.
        train (bool): True if extracting features for the training set, False for the test set.
        model_type (ModelType): Enum representing the type of pretrained model to use.
        dataset_dir (str): Path to the directory containing the dataset.
        model_dir (str): Path to the directory containing the model cache.
        feature_dir (str): Path to the directory where extracted features will be saved.
        dataloader (Callable[..., DataLoader[Any]]): Callable to create DataLoader for the dataset.
        trainer (pl.Trainer): PyTorch Lightning Trainer for feature extraction.
    """
    split = "train" if train else "test"
    dst_dir = os.path.join(feature_dir, f"{dataset_type.name}")
    vector_file = os.path.join(dst_dir, f"{model_type.name}.hdf")
    os.makedirs(dst_dir, exist_ok=True)

    if check_existing_features(vector_file, split):
        print(
            f"{split} features have already been computed for the {dataset_type.name}"
            + f" dataset with the {model_type.name} model"
        )
        return  # skip feature extraction

    model, transform = create_model(model_type, cache_dir=model_dir)
    dataset = create_dataset(
        dataset_type, root=dataset_dir, train=train, transform=transform
    )

    model = BackboneWrapper(model)
    preds = trainer.predict(model, dataloader(dataset))

    feature_list, label_list = zip(*preds)  # type: ignore[misc]
    features = np.concatenate(feature_list)
    labels = np.concatenate(label_list)

    save_vectors(features, labels, vector_file, split)
