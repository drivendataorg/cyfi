from pathlib import Path

import numpy as np

from cyano.data.features import generate_features
from cyano.data.satellite_data import download_satellite_data
from cyano.data.utils import add_unique_identifier

ASSETS_DIR = Path(__file__).parent / "assets"


def test_known_features(train_data, features_config):
    train_data = add_unique_identifier(train_data)

    # Generate features based on saved imagery
    features = generate_features(
        train_data,
        features_config,
        cache_dir=str(ASSETS_DIR / "feature_cache"),
    )

    # Check that generated stats match known imagery stats
    assert np.isclose(features.loc["3a2c48812b551d720f8d56772efa6df1", "B02_mean"], 161.532712)
    assert np.isclose(features.loc["3a2c48812b551d720f8d56772efa6df1", "B02_min"], 50)
    assert np.isclose(features.loc["3a2c48812b551d720f8d56772efa6df1", "B02_max"], 1182)


def test_download_satellite_data(tmp_path, satellite_meta, train_data, features_config):
    # Download imagery
    features_config.use_sentinel_bands = ["B02", "B03"]
    train_data = add_unique_identifier(train_data)
    download_satellite_data(satellite_meta, train_data, features_config, tmp_path)

    # Sentinel image cache directory exists
    sentinel_dir = tmp_path / f"sentinel_{features_config.image_feature_meter_window}"
    assert sentinel_dir.exists()
    assert len(list(sentinel_dir.rglob("*.npy"))) > 0

    # Check that the structure of saved image arrays is correct
    for sample_dir in sentinel_dir.iterdir():
        # Correct number of items per sample
        sample_item_dirs = list(sample_dir.iterdir())
        assert len(sample_item_dirs) == features_config.n_sentinel_items

        # Correct bands for each item
        for sample_item_dir in sample_item_dirs:
            assert set([pth.stem for pth in sample_item_dir.iterdir()]) == set(
                features_config.use_sentinel_bands
            )
