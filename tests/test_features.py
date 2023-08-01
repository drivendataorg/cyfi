from pathlib import Path

import numpy as np

from cyano.data.features import generate_features

ASSETS_DIR = Path(__file__).parent / "assets"


def test_known_features(train_data, train_config):
    # Generate features based on saved imagery
    train_config.features_config.cache_dir = str(ASSETS_DIR / "feature_cache")
    features = generate_features(
        train_data.set_index("uid").loc[["ofhd", "rszn"]], train_config.features_config
    )

    # Check that generated stats match known imagery stats
    assert np.isclose(features.loc["ofhd", "B02_mean"], 161.532712)
    assert np.isclose(features.loc["ofhd", "B02_min"], 50)
    assert np.isclose(features.loc["ofhd", "B02_max"], 1182)
