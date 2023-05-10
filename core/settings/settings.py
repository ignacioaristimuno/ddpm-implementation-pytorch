from core.base_objects import DiffusionModelType
from typing import Any, Dict
from yaml import safe_load


CONFIG_DIR = "core/settings/config.yml"


def get_config(key: str = None) -> Dict[str, Any]:
    """Function for fetching the default configs a certian key"""
    with open(CONFIG_DIR, "r") as file:
        config = safe_load(file)
    return config[key] if key else config


def get_model_config(model_type: DiffusionModelType) -> Dict[str, Any]:
    """Function for fetching the default configs of each model"""
    with open(CONFIG_DIR, "r") as file:
        return safe_load(file)["Models"][model_type.value]
