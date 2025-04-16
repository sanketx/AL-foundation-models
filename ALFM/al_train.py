"""Script to run the active learning loop."""

import logging
import os

import h5py
import hydra
from dotenv import dotenv_values
from omegaconf import DictConfig

from ALFM.src.run.al_train import al_train


def get_vector_file(dataset: str, model: str, feature_dir: str) -> str:
    vector_file = f"{os.path.join(feature_dir, dataset, model)}.hdf"

    if not os.path.exists(vector_file):
        raise FileNotFoundError(f"File {vector_file} does not exist.")

    with h5py.File(vector_file) as fh:
        keys = set(fh.keys())

    if keys != {"train", "test"}:
        missing = set(["train", "test"]) - keys
        raise KeyError(f"Missing features for the {missing.pop()} split")

    return vector_file


@hydra.main(
    config_path="conf",
    config_name="al_train.yaml",
    version_base="1.1",
)
def main(cfg: DictConfig) -> None:
    env = dotenv_values()
    feature_dir = env.get("FEATURE_CACHE_DIR", None)
    log_dir = env.get("LOG_DIR", None)

    # if cfg.dataset.name in ["cifar100", "food101", "imagenet100", "domainnetreal"]:
    #     log_dir = "/sdf/home/s/sanketg/projects/AL-foundation-models/ALFM/supp_logs"

    assert (
        feature_dir is not None
    ), "Please set the 'FEATURE_CACHE_DIR' variable in your .env file"

    assert log_dir is not None, "Please set the 'LOG_DIR' variable in your .env file"

    vector_file = get_vector_file(cfg.dataset.name, cfg.model.name, feature_dir)

    try:
        al_train(vector_file, log_dir, cfg)
    except Exception as err:
        logging.error(f"Error: {err}")


if __name__ == "__main__":
    main()
