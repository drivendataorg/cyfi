import yaml
from pathlib import Path

import pandas as pd
import pytest

from cyano.config import TrainConfig, PredictConfig, FeaturesConfig, ModelConfig

ASSETS_DIR = Path(__file__).parent / "assets"


@pytest.fixture(scope="session")
def train_data_path() -> Path:
    return ASSETS_DIR / "train_data.csv"


@pytest.fixture(scope="session")
def train_data(train_data_path) -> pd.DataFrame:
    return pd.read_csv(train_data_path)


@pytest.fixture(scope="session")
def predict_data_path() -> Path:
    return ASSETS_DIR / "predict_data.csv"


@pytest.fixture(scope="session")
def predict_data(predict_data_path) -> pd.DataFrame:
    return pd.read_csv(predict_data_path)


@pytest.fixture
def train_config(tmp_path_factory):
    return TrainConfig(
        features_config=FeaturesConfig(
            use_sentinel_bands=["B02"],
            image_feature_meter_window=500,
            satellite_features=["B02_mean", "B02_min", "B02_max"],
        ),
        tree_model_config=ModelConfig(save_dir=str(tmp_path_factory.mktemp("model_dir"))),
    )


@pytest.fixture
def predict_config(tmp_path_factory):
    with (ASSETS_DIR / "trained_model" / "config.yaml").open("r") as f:
        config = yaml.safe_load(f)

    # TODO: probably want a function for this
    config["features_config"]["cache_dir"] = None
    return PredictConfig(
        features_config=config["features_config"],
        preds_path=str(tmp_path_factory.mktemp("test_predict") / "preds.csv"),
        weights="tests/assets/trained_model/lgb_model.txt",
    )
