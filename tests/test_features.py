from pathlib import Path

import numpy as np

from cyano.data.features import generate_features
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
    assert np.isclose(features.loc["129eb14803", "B02_mean"], 161.532712)
    assert np.isclose(features.loc["129eb14803", "B02_min"], 50)
    assert np.isclose(features.loc["129eb14803", "B02_max"], 1182)
