from pathlib import Path

import pandas as pd
import pytest

from cyano.config import FeaturesConfig
from cyano.experiment.experiment import ExperimentConfig

ASSETS_DIR = Path(__file__).parent / "assets"


@pytest.fixture(scope="session")
def train_data_path() -> Path:
    return ASSETS_DIR / "train_data.csv"


@pytest.fixture(scope="session")
def train_data(train_data_path) -> pd.DataFrame:
    return pd.read_csv(train_data_path)


@pytest.fixture(scope="session")
def satellite_meta() -> pd.DataFrame:
    return pd.read_csv(ASSETS_DIR / "satellite_metadata.csv")


@pytest.fixture(scope="session")
def experiment_config_path() -> Path:
    return ASSETS_DIR / "experiment_config.yaml"


@pytest.fixture(scope="session")
def predict_data_path() -> Path:
    return ASSETS_DIR / "predict_data.csv"


@pytest.fixture(scope="session")
def predict_data(predict_data_path) -> pd.DataFrame:
    return pd.read_csv(predict_data_path)


@pytest.fixture
def features_config():
    return FeaturesConfig(
        use_sentinel_bands=["B02"],
        image_feature_meter_window=500,
        satellite_image_features=["B02_mean", "B02_min", "B02_max"],
        climate_features=["TMP_min", "SPFH_mean"],
        climate_variables=["TMP", "SPFH"],
        climate_level="2 m above ground",
    )


@pytest.fixture
def experiment_config():
    return ExperimentConfig.from_file(ASSETS_DIR / "experiment_config.yaml")
