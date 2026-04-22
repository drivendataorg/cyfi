from distutils.util import strtobool
import os
from pathlib import Path
import sys

from loguru import logger
import numpy as np
import pytest

from cyfi.config import FeaturesConfig
from cyfi.data.features import generate_all_features, calculate_metadata_features
from cyfi.data.satellite_data import download_satellite_data, generate_candidate_metadata
from cyfi.data.utils import add_unique_identifier

ASSETS_DIR = Path(__file__).parent / "assets"

IN_GITHUB_ACTIONS = strtobool(os.getenv("GITHUB_ACTIONS", "false"))


def test_known_features(train_data, features_config, satellite_meta):
    train_data = add_unique_identifier(train_data)

    # Generate features based on saved imagery
    sentinel_meta, features = generate_all_features(
        train_data,
        satellite_meta,
        features_config,
        cache_dir=str(ASSETS_DIR / "feature_cache"),
    )

    # Check that generated stats match known imagery stats
    assert np.isclose(features.loc["3a2c48812b551d720f8d56772efa6df1", "B02_mean"], 402.2583)
    assert np.isclose(features.loc["3a2c48812b551d720f8d56772efa6df1", "B02_min"], 309)
    assert np.isclose(features.loc["3a2c48812b551d720f8d56772efa6df1", "B02_max"], 1296)

    # Check expected columns in sentinel metadata (cloud_pct and num_water_pixels are not included because features_config doesn't use these)
    assert (sentinel_meta.columns == ["item_id", "days_before_sample", "visual_href"]).all()


def test_generate_candidate_metadata(mocker, train_data, features_config):
    train_data = add_unique_identifier(train_data)

    # Mock the STAC search results
    mock_search = mocker.Mock()
    
    # Create mock items that match what the test expects
    mock_item_ids = [
        "S2A_MSIL2A_20190824T154911_R054_T18TVL_20201106T052956",
        "S2B_MSIL2A_20190819T154819_R054_T18TVL_20201005T022720",
        "S2A_MSIL2A_20190814T154911_R054_T18TVL_20201005T001501",
        "S2B_MSIL2A_20190809T154819_R054_T18TVL_20201004T222827",
        "S2A_MSIL2A_20190804T154911_R054_T18TVL_20201004T201836",
        "S2B_MSIL2A_20190730T154819_R054_T18TVL_20201005T200628",
        "S2A_MSIL2A_20170728T155901_R097_T17SPV_20210210T154351"
    ]
    
    mock_items = []
    for item_id in mock_item_ids:
        item = mocker.Mock()
        item.id = item_id
        # Extract date from ID for mock properties
        date_str = item_id.split("_")[2][:8]
        item.properties = {"datetime": f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}T00:00:00Z"}
        item.assets = {"rendered_preview": mocker.Mock(href=f"https://example.com/{item_id}.jpg")}
        mock_items.append(item)
        
    mock_search.item_collection.return_value = mock_items
    mocker.patch("cyfi.data.satellite_data.search_planetary_computer", return_value=mock_search)

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

    # Check that candidate metadata matches expected values for our mock
    assert candidate_meta.item_id.is_unique
    assert len(candidate_meta) == len(mock_item_ids)
    assert (
        "S2A_MSIL2A_20170728T155901_R097_T17SPV_20210210T154351" in candidate_meta.item_id.values
    )
    assert (
        "S2B_MSIL2A_20190819T154819_R054_T18TVL_20201005T022720" in candidate_meta.item_id.values
    )

    assert "visual_href" in candidate_meta.columns


def test_download_satellite_data(mocker, tmp_path, satellite_meta, train_data, features_config, capsys):
    features_config.use_sentinel_bands = ["B02", "B03"]
    train_data = add_unique_identifier(train_data)

    # Mock the asset downloader to just create dummy files
    def mock_download_assets(item_id, assets, bbox, sample_dir, features_config):
        item_dir = sample_dir / item_id
        item_dir.mkdir(parents=True, exist_ok=True)
        for band in features_config.use_sentinel_bands:
            (item_dir / f"{band}.npy").touch()
        return True

    mocker.patch("cyfi.data.satellite_data._download_item_assets", side_effect=mock_download_assets)

    # Test case when nothing is downloaded, and download_row errors for every item
    # We mock _download_item_assets to return False to simulate failure
    mocker.patch("cyfi.data.satellite_data._download_item_assets", return_value=False)
    
    new_satellite_meta = satellite_meta.copy()
    new_satellite_meta["B02_href"] = "bad-href"
    with pytest.raises(
        ValueError,
        match="No satellite imagery was successfully downloaded. Check the per-item debug logs for details.",
    ):
        download_satellite_data(new_satellite_meta, train_data, features_config, tmp_path)

    # Log to stdout so we can check the results with capsys
    logger.add(sys.stdout, level="DEBUG")

    # Test case when some items are downloaded, but not all
    # Mock to fail only for the first item
    mocker.patch("cyfi.data.satellite_data._download_item_assets", side_effect=[False] + [True] * 100)
    
    new_satellite_meta = satellite_meta.copy()
    new_satellite_meta.loc[0, "B02_href"] = "bad-href"
    download_satellite_data(new_satellite_meta, train_data, features_config, tmp_path)
    captured = capsys.readouterr()
    assert "SUCCESS" not in captured.out
    assert "WARNING" in captured.out
    assert "item(s) could not be downloaded." in captured.out

    # Test case when all imagery is downloaded successfully
    mocker.patch("cyfi.data.satellite_data._download_item_assets", side_effect=mock_download_assets)
    download_satellite_data(satellite_meta, train_data, features_config, tmp_path)

    # Check that logged message includes a success and expected text
    captured = capsys.readouterr()
    assert "SUCCESS" in captured.out
    assert "Downloaded all satellite imagery successfully." in captured.out

    # Check that Sentinel image cache directory exists
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


@pytest.mark.network
@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Entails ~2GB download of land cover map")
def test_land_cover_features(train_data):
    feature_config = FeaturesConfig(sample_meta_features=["land_cover"])
    train_data = add_unique_identifier(train_data)
    features = calculate_metadata_features(
        train_data,
        feature_config,
    )

    assert features.land_cover.notna().all()
    # Check that generated land cover classes match known classes
    assert features.loc["9c601f226c2af07d570134127a7fda27", "land_cover"] == 90
    assert features.loc["3a2c48812b551d720f8d56772efa6df1", "land_cover"] == 70
    assert features.shape[0] == train_data.shape[0]
