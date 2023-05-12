"""Script to extract and save image features using pretrained backbones."""

import os

import hydra
from dotenv import dotenv_values
from omegaconf import DictConfig

from ALFM.src.run.feature_extraction import extract_features


os.environ["SLURM_JOB_NAME"] = "interactive"


@hydra.main(
    config_path="conf",
    config_name="feature_extraction.yaml",
    version_base="1.1",
)
def main(cfg: DictConfig) -> None:
    """Extract and save image features using pretrained backbones.

    This script uses the `hydra` library to manage configuration and the
    `omegaconf` library to access the configuration. The script extracts image
    features using a specified dataset and a specified pretrained model. The
    dataset and model are specified in a YAML configuration file. The script
    uses environment variables to determine the directory paths for the dataset
    and model cache.

    Attributes:
        dataset_dir (str): Path to the directory containing the dataset.
        model_dir (str): Path to the directory containing the model cache.
        feature_dir (str): Path to the directory containing the features

    Raises:
        ValueError: If an invalid split is specified.
        AssertionError: If any of the 'DATASET_DIR', 'MODEL_CACHE_DIR', or
        'FEATURE_CACHE_DIR' environment variables are not set.
    """
    dataloader = hydra.utils.instantiate(cfg.dataloader)

    env = dotenv_values()
    dataset_dir = env.get("DATASET_DIR", None)
    model_dir = env.get("MODEL_CACHE_DIR", None)
    feature_dir = env.get("FEATURE_CACHE_DIR", None)

    assert (
        dataset_dir is not None
    ), "Please set the 'DATASET_DIR' variable in your .env file"

    assert (
        model_dir is not None
    ), "Please set the 'MODEL_CACHE_DIR' variable in your .env file"

    assert (
        feature_dir is not None
    ), "Please set the 'FEATURE_CACHE_DIR' variable in your .env file"

    if cfg.split not in ["train", "test", "both"]:
        raise ValueError(
            f"Invalid split: '{cfg.split}'. Please specify a valid split: 'train' | 'test' | 'both'"
        )

    if cfg.split in ["train", "both"]:
        extract_features(
            cfg.dataset,
            True,
            cfg.model,
            dataset_dir,
            model_dir,
            feature_dir,
            dataloader,
            cfg.trainer,
        )

    if cfg.split in ["test", "both"]:
        extract_features(
            cfg.dataset,
            False,
            cfg.model,
            dataset_dir,
            model_dir,
            feature_dir,
            dataloader,
            cfg.trainer,
        )


if __name__ == "__main__":
    main()
