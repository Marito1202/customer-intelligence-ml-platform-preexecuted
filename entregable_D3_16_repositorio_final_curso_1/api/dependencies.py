from functools import lru_cache
from pathlib import Path

import yaml

from src.models.prediction import load_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"


@lru_cache(maxsize=1)
def load_params() -> dict:
    with PARAMS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


@lru_cache(maxsize=1)
def get_model():
    params = load_params()

    model_path = (
        PROJECT_ROOT
        / params["paths"]["model"]
    )

    return load_model(model_path)


def get_prediction_threshold() -> float:
    params = load_params()

    return float(
        params["prediction"]["threshold"]
    )