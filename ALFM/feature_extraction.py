"""Script to extract and save image features using pretrained backbones."""

import hydra
from dotenv import dotenv_values
from omegaconf import DictConfig

from ALFM.src.datasets.registry import DatasetType
from ALFM.src.models.registry import ModelType
from ALFM.src.run.feature_extraction import extract_features


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
        dataset_type (DatasetType): Enum representing the type of dataset to use.
        model_type (ModelType): Enum representing the type of pretrained model to use.
        dataset_dir (str): Path to the directory containing the dataset.
        model_dir (str): Path to the directory containing the model cache.

    Raises:
        ValueError: If an invalid model, dataset. or split is specified.
        AssertionError: If any of the 'DATASET_DIR', 'MODEL_CACHE_DIR', or
        'FEATURE_CACHE_DIR' environment variables are not set.
    """
    dataset_type = DatasetType[cfg.dataset.name]
    model_type = ModelType[cfg.model.name]
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
            dataset_type,
            True,
            model_type,
            dataset_dir,
            model_dir,
            feature_dir,
            dataloader,
            hydra.utils.instantiate(cfg.trainer),
        )
    if cfg.split in ["test", "both"]:
        extract_features(
            dataset_type,
            False,
            model_type,
            dataset_dir,
            model_dir,
            feature_dir,
            dataloader,
            hydra.utils.instantiate(cfg.trainer),
        )


if __name__ == "__main__":
    main()
