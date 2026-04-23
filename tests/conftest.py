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
def experiment_config(experiment_config_path):
    return ExperimentConfig.from_file(experiment_config_path)


@pytest.fixture
def mock_satellite_data(mocker, satellite_meta):
    def _identify_satellite_data(samples, config):
        # Filter satellite_meta to samples in samples
        found = satellite_meta[satellite_meta.sample_id.isin(samples.index)]
        missing_ids = set(samples.index) - set(found.sample_id)

        # Intentionally keep one sample missing for tests that expect it
        # (e.g., test_cli_predict in test_cli.py)
        missing_ids.discard("e66ea0c31ba500d5d4ac4c610b8cf508")

        if missing_ids:
            dummy_row = satellite_meta.iloc[0].copy()
            dummies = []
            for sid in missing_ids:
                d = dummy_row.copy()
                d["sample_id"] = sid
                dummies.append(d)
            found = pd.concat([found, pd.DataFrame(dummies)])
        return found

    def _download_satellite_data(satellite_meta, samples, config, cache_dir):
        imagery_dir = Path(cache_dir) / f"sentinel_{config.image_feature_meter_window}"
        for _, row in satellite_meta.iterrows():
            sample_item_dir = imagery_dir / f"{row.sample_id}/{row.item_id}"
            sample_item_dir.mkdir(parents=True, exist_ok=True)
            for band in config.use_sentinel_bands:
                (sample_item_dir / f"{band}.npy").touch()
        return len(satellite_meta)

    mocker.patch(
        "cyfi.pipeline.identify_satellite_data", side_effect=_identify_satellite_data
    )
    mocker.patch(
        "cyfi.pipeline.download_satellite_data", side_effect=_download_satellite_data
    )


@pytest.fixture
def mock_metadata_features(mocker):
    def _calculate_metadata_features(samples, config):
        # Return a copy of samples with the requested metadata columns
        # For simplicity, we just return dummy values for any requested feature
        features = samples.copy()
        for feature in config.sample_meta_features:
            if feature not in features.columns:
                features[feature] = 0
        return features[config.sample_meta_features]

    mocker.patch(
        "cyfi.data.features.calculate_metadata_features",
        side_effect=_calculate_metadata_features,
    )


def pytest_addoption(parser):
    parser.addoption(
        "--run-network", action="store_true", default=False, help="run tests that require network"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-network"):
        # --run-network given in cli: do not skip network tests
        return
    skip_network = pytest.mark.skip(reason="need --run-network option to run")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip_network)

# Mock API fixtures for Issue #71
import json
from pathlib import Path
from unittest.mock import Mock, patch

@pytest.fixture
def mock_stac_search_response():
    """Load mock STAC search response."""
    mock_file = Path(__file__).parent / "fixtures/mock_responses/stac_search_response.json"
    with open(mock_file) as f:
        return json.load(f)

@pytest.fixture
def mock_planetary_computer(mock_stac_search_response):
    """Mock the Planetary Computer API client."""
    with patch('cyfi.data.satellite_data.catalog') as mock_catalog:
        # Create mock search results
        mock_search = Mock()
        mock_item = Mock()
        mock_item.to_dict.return_value = mock_stac_search_response['items'][0]
        mock_item.id = mock_stac_search_response['items'][0]['id']
        mock_item.properties = mock_stac_search_response['items'][0]['properties']
        mock_item.assets = mock_stac_search_response['items'][0]['assets']
        
        mock_search.item_collection.return_value = [mock_item]
        mock_catalog.search.return_value = mock_search
        
        # Mock the signing function
        with patch('planetary_computer.sign_inplace') as mock_sign:
            mock_sign.side_effect = lambda x: x  # Identity function
            yield mock_catalog

@pytest.fixture
def mock_pc_sign():
    """Mock planetary_computer.sign function."""
    with patch('planetary_computer.sign') as mock_sign:
        mock_sign.side_effect = lambda url: url
        yield mock_sign

@pytest.fixture
def mock_download_satellite_data():
    """Mock the download_satellite_data function to avoid real downloads."""
    with patch('cyfi.data.satellite_data.download_satellite_data') as mock_download:
        mock_download.return_value = 1  # Return successful downloads count
        yield mock_download
