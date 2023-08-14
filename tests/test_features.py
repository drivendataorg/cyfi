from pathlib import Path

import numpy as np

from cyano.data.features import generate_features
from cyano.data.satellite_data import download_satellite_data, generate_candidate_metadata
from cyano.data.utils import add_unique_identifier

ASSETS_DIR = Path(__file__).parent / "assets"


def test_known_features(train_data, features_config, satellite_meta):
    train_data = add_unique_identifier(train_data)

    # Generate features based on saved imagery
    features = generate_features(
        train_data,
        satellite_meta,
        features_config,
        cache_dir=str(ASSETS_DIR / "feature_cache"),
    )

    # Check that generated stats match known imagery stats
    assert np.isclose(features.loc["3a2c48812b551d720f8d56772efa6df1", "B02_mean"], 402.2583)
    assert np.isclose(features.loc["3a2c48812b551d720f8d56772efa6df1", "B02_min"], 309)
    assert np.isclose(features.loc["3a2c48812b551d720f8d56772efa6df1", "B02_max"], 1296)


def test_generate_candidate_metadata(train_data, features_config):
    train_data = add_unique_identifier(train_data)

    candidate_meta, sample_item_map = generate_candidate_metadata(train_data, features_config)

    # Check that item map has the correct samples and matches known values
    assert len(sample_item_map) == len(train_data)
    assert set(sample_item_map.keys()) == set(train_data.index)
    assert sample_item_map["3a2c48812b551d720f8d56772efa6df1"]["sentinel_item_ids"] == [
        "S2A_MSIL2A_20190824T154911_R054_T18TVL_20201106T052956",
        "S2B_MSIL2A_20190819T154819_R054_T18TVL_20201005T022720",
        "S2A_MSIL2A_20190814T154911_R054_T18TVL_20201005T001501",
        "S2B_MSIL2A_20190809T154819_R054_T18TVL_20201004T222827",
        "S2A_MSIL2A_20190804T154911_R054_T18TVL_20201004T201836",
        "S2B_MSIL2A_20190730T154819_R054_T18TVL_20201005T200628",
    ]

    # Check that candidate metadata matches known expected values
    assert candidate_meta.item_id.is_unique
    assert len(candidate_meta) == 9
    assert (
        "S2A_MSIL2A_20170728T155901_R097_T17SPV_20210210T154351" in candidate_meta.item_id.values
    )
    assert (
        "S2B_MSIL2A_20190819T154819_R054_T18TVL_20201005T022720" in candidate_meta.item_id.values
    )


def test_download_satellite_data(tmp_path, satellite_meta, train_data, features_config):
    # Download imagery
    features_config.use_sentinel_bands = ["B02", "B03"]
    train_data = add_unique_identifier(train_data)
    download_satellite_data(satellite_meta, train_data, features_config, tmp_path, num_processes=4)

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
