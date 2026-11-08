from pathlib import Path
import yaml


def load_config(path="configs/config.yaml"):
    with open(Path(path), "r") as f:
        return yaml.safe_load(f)
