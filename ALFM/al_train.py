"""Script to run the active learning loop."""

import os

import h5py
import hydra
from ALFM.src.run.al_train import al_train
from dotenv import dotenv_values
from omegaconf import DictConfig


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

    assert (
        feature_dir is not None
    ), "Please set the 'FEATURE_CACHE_DIR' variable in your .env file"

    vector_file = get_vector_file(cfg.dataset.name, cfg.model.name, feature_dir)
    al_train(vector_file, cfg)


if __name__ == "__main__":
    main()
