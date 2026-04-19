
import yaml
from pathlib import Path


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
    return params