from pathlib import Path

import pandas as pd
import pytest

from cyfi.config import FeaturesConfig
from cyfi.experiment import ExperimentConfig

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
def experiment_config_with_folds_path() -> Path:
    return ASSETS_DIR / "experiment_config_with_folds.yaml"


@pytest.fixture(scope="session")
def predict_data_path() -> Path:
    return ASSETS_DIR / "predict_data.csv"


@pytest.fixture(scope="session")
def predict_data(predict_data_path) -> pd.DataFrame:
    return pd.read_csv(predict_data_path)


@pytest.fixture(scope="session")
def evaluate_data_path() -> Path:
    return ASSETS_DIR / "evaluate_data.csv"


@pytest.fixture(scope="session")
def evaluate_data_features() -> pd.DataFrame:
    return pd.read_csv(ASSETS_DIR / "experiment" / "features_test.csv", index_col=0)


@pytest.fixture(scope="session")
def local_model_path() -> Path:
    return ASSETS_DIR / "experiment" / "model.zip"


@pytest.fixture
def features_config():
    return FeaturesConfig(
        use_sentinel_bands=["B02"],
        image_feature_meter_window=500,
        satellite_image_features=["B02_mean", "B02_min", "B02_max"],
        pc_days_search_window=30,
        pc_meters_search_window=1000,
        n_sentinel_items=1,
        satellite_meta_features=[],
        sample_meta_features=[],
        filter_to_water_area=False,
        max_cloud_percent=None,
    )


@pytest.fixture
def experiment_config():
    return ExperimentConfig.from_file(ASSETS_DIR / "experiment_config.yaml")
