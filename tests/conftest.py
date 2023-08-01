from pathlib import Path

import pandas as pd
import pytest

from cyano.config import TrainConfig, PredictConfig, FeaturesConfig, ModelConfig

ASSETS_DIR = Path(__file__).parent / "assets"


@pytest.fixture(scope="session")
def train_data() -> pd.DataFrame:
    return pd.read_csv(ASSETS_DIR / "train_data.csv")

@pytest.fixture(scope="session")
def predict_data() -> pd.DataFrame:
    return pd.read_csv(ASSETS_DIR / "train_data.csv")[["date", "latitude", "longitude"]]

@pytest.fixture(scope="session")
def train_config(tmp_path_factory):
    return TrainConfig(
        features_config=FeaturesConfig(
            use_sentinel_bands=["B02"],
            image_feature_meter_window=500,
            satellite_features=["B02_mean", "B02_min", "B02_max"],
        ),
        model_config=ModelConfig(save_dir=str(tmp_path_factory.mktemp("model_dir")))
    )

@pytest.fixture(scope="session")
def predict_config(train_config):
    return PredictConfig(trained_model_dir=str(ASSETS_DIR / "trained_model"))