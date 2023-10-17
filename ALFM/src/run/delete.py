import csv
from pathlib import Path

from dotenv import dotenv_values
from omegaconf import DictConfig
from omegaconf import OmegaConf
from rich.progress import track


def my_app() -> None:
    cfg = OmegaConf.from_cli()
    log_dir = dotenv_values().get("LOG_DIR", None)
    assert log_dir is not None, "Please set the 'LOG_DIR' variable in your .env file"

    # Directories for configuration and results
    config_dir = Path(log_dir) / "configs"
    results_dir = Path(log_dir) / "results"

    # Extract the delete flag and remove it from cfg
    OmegaConf.set_struct(cfg, False)
    delete_flag = cfg.pop("delete", False)

    # Get list of all config files
    for config_file in track(
        config_dir.glob("**/*.yaml"), description="[green]Analyzing"
    ):
        # Construct path to the results file
        relative_path = config_file.relative_to(config_dir)
        results_file = results_dir / relative_path.with_suffix(".csv")

        # Load the YAML file with OmegaConf
        conf = OmegaConf.load(config_file)
        merged_conf = OmegaConf.merge(conf, cfg)

        # Check if the subset of parameters from the config file matches the command line parameters
        if merged_conf == conf:
            # If conditions are met, check if results file exists
            if results_file.is_file():
                # Check if results file is complete
                with open(results_file, "r") as f:
                    reader = csv.reader(f)
                    lines = list(reader)
                    if len(lines) != 21:
                        # Incomplete results file found, print and possibly delete
                        print(f"Incomplete: {len(lines)} {relative_path}, ")
                        if delete_flag:
                            config_file.unlink()
                            results_file.unlink()
            else:
                # No results file found, print and possibly delete
                print(f"Missing: {relative_path}")
                if delete_flag:
                    config_file.unlink()


if __name__ == "__main__":
    my_app()
