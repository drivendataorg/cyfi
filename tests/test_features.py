import os
from pathlib import Path
import sys

from loguru import logger
import numpy as np
from pystac import STACError
from pystac_client.exceptions import APIError
import pytest
import rasterio

from cyfi.config import FeaturesConfig
from cyfi.data.features import generate_all_features, calculate_metadata_features
from cyfi.data.satellite_data import (
    download_row,
    download_satellite_data,
    generate_candidate_metadata,
    retry_stac,
)
from cyfi.data.utils import add_unique_identifier

ASSETS_DIR = Path(__file__).parent / "assets"

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


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
    assert len(candidate_meta) == 34
    assert (
        "S2A_MSIL2A_20170728T155901_R097_T17SPV_20210210T154351" in candidate_meta.item_id.values
    )
    assert (
        "S2B_MSIL2A_20190819T154819_R054_T18TVL_20201005T022720" in candidate_meta.item_id.values
    )

    assert "visual_href" in candidate_meta.columns


def test_download_satellite_data(tmp_path, satellite_meta, train_data, features_config, capsys):
    features_config.use_sentinel_bands = ["B02", "B03"]
    train_data = add_unique_identifier(train_data)

    # Test case when nothing is downloaded, and download_row errors for every item
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
    new_satellite_meta = satellite_meta.copy()
    new_satellite_meta.loc[0, "B02_href"] = "bad-href"
    download_satellite_data(new_satellite_meta, train_data, features_config, tmp_path)
    captured = capsys.readouterr()
    assert "SUCCESS" not in captured.out
    assert "WARNING" in captured.out
    assert "item(s) could not be downloaded." in captured.out

    # Test case when all imagery is downloaded successfully
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


def test_download_row_closes_file(tmp_path, satellite_meta, train_data, features_config, mocker):
    rasterio_open = mocker.spy(rasterio, "open")
    samples = add_unique_identifier(train_data)
    assert download_row((0, satellite_meta.iloc[0]), samples, tmp_path, features_config)

    assert rasterio_open.call_count == 1
    assert rasterio_open.spy_return.closed


def test_retry_stac(mocker):
    sleep = mocker.patch("cyfi.data.satellite_data.time.sleep")

    # Test case when the API fails twice and then succeeds
    func = mocker.Mock(side_effect=[STACError("down"), APIError("down"), "ok"])
    assert retry_stac(func, "arg", key="value") == "ok"
    assert func.call_count == 3
    func.assert_called_with("arg", key="value")
    assert sleep.call_args_list == [mocker.call(2), mocker.call(4)]

    # Test case when the error is not from the API, which should not be retried
    func = mocker.Mock(side_effect=ValueError("bad input"))
    with pytest.raises(ValueError):
        retry_stac(func)
    assert func.call_count == 1

    # Test case when the API never recovers
    func = mocker.Mock(side_effect=STACError("down"))
    with pytest.raises(STACError, match="down"):
        retry_stac(func, attempts=3)
    assert func.call_count == 3


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


def cloud_filter_config():
    """Features config where the first item in satellite_meta is too cloudy and the second
    is clear. Their cloud fractions in the sample bounding boxes are 0.0041 and 0.0012, so
    max_cloud_percent (a fraction despite its name) is set between them.
    """
    return FeaturesConfig(
        use_sentinel_bands=["B02"],
        image_feature_meter_window=500,
        satellite_image_features=["B02_mean", "B02_min", "B02_max"],
        satellite_meta_features=[],
        sample_meta_features=[],
        filter_to_water_area=False,
        max_cloud_percent=0.002,
    )


def test_download_satellite_data_skips_cloudy_items(tmp_path, satellite_meta, train_data):
    config = cloud_filter_config()
    train_data = add_unique_identifier(train_data)
    cloudy, clear = satellite_meta.iloc[0], satellite_meta.iloc[1]

    # Only the SCL band is downloaded for the cloudy item
    assert download_satellite_data(satellite_meta, train_data, config, tmp_path) == 2
    img_dir = tmp_path / "sentinel_500"
    cloudy_files = {pth.name for pth in (img_dir / cloudy.sample_id / cloudy.item_id).iterdir()}
    clear_files = {pth.name for pth in (img_dir / clear.sample_id / clear.item_id).iterdir()}
    assert cloudy_files == {"SCL.npy"}
    assert clear_files == {"B02.npy", "SCL.npy"}

    _, features = generate_all_features(train_data, satellite_meta, config, tmp_path)
    assert features.index.tolist() == [clear.sample_id]
