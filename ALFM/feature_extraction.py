import hydra
from dotenv import load_dotenv
from hydra_zen import instantiate
from omegaconf import DictConfig


load_dotenv()


@hydra.main(
    config_path="conf", config_name="feature_extraction.yaml", version_base="1.1"
)
def main(cfg: DictConfig) -> None:
    dataset = instantiate(cfg.dataset_cfg)
    print(dataset)


if __name__ == "__main__":
    main()
