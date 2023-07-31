from pathlib import Path

import pandas as pd
import pytest

ASSETS_DIR = Path(__file__).parent / "assets"


@pytest.fixture(scope="session")
def train_data() -> pd.DataFrame:
    return pd.read_csv(ASSETS_DIR / "train_data.csv")

@pytest.fixture(scope="session")
def predict_data() -> pd.DataFrame:
    return pd.read_csv(ASSETS_DIR / "train_data.csv")[["date", "latitude", "longitude"]]

@pytest.fixture(scope="session")
def train_config():
    # TODO: should be pydantic model
    return {
        "use_sentinel_bands": ["B02"],
        "image_feature_meter_window": 500,
        "satellite_features": ["B02_mean", "B02_min", "B02_max"],
    }

@pytest.fixture(scope="session")
def predict_config(train_config):
    # TODO: shouldn't include all the train config params
    config = train_config.copy()
    config["cyano_model_config"] = {"trained_model_dir": str(ASSETS_DIR / "trained_model")}
    return config