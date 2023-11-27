## Code to generate features from raw downloaded source data
import functools
import tarfile
from typing import Union

from cloudpathlib import S3Client
import cv2
from loguru import logger
import numpy as np
import pandas as pd
from pathlib import Path
import platformdirs
from scipy.stats.mstats import winsorize
from tqdm.contrib.concurrent import process_map
import xarray as xr


from cyfi.config import FeaturesConfig, SATELLITE_FEATURE_CALCULATORS


def calculate_satellite_features(
    satellite_meta: pd.DataFrame,
    config: FeaturesConfig,
    cache_dir: Union[str, Path],
) -> pd.DataFrame:
    """Generate features from satellite data

    Args:
        satellite_meta (pd.DataFrame): Dataframe of satellite metadata
            for all pystac items that have been selected for use in
            feature generation
        config (FeaturesConfig): Configuration, including
            directory where raw source data is saved
        cache_dir (Union[str, Path]): Cache directory where raw imagery
            is saved

    Returns:
        pd.DataFrame: Dataframe where the index is sample ID and there is one column
            for each satellite feature. There will only be rows for samples with
            satellite imagery. Each row is a unique combination of sample ID and
            item ID
    """
    # Calculate satellite metadata features
    if "month" in config.satellite_meta_features:
        satellite_meta["month"] = pd.to_datetime(satellite_meta.datetime).dt.month

    # Iterate over selected sample / item combinations
    logger.info(f"Generating satellite features for {satellite_meta.shape[0]:,} images.")
    satellite_features = process_map(
        functools.partial(
            _calculate_satellite_features_for_sample_item, config=config, cache_dir=cache_dir
        ),
        satellite_meta.sample_id,
        satellite_meta.item_id,
        chunksize=1,
        total=len(satellite_meta),
        # Only log progress bar if debug message is logged
        disable=(logger._core.min_level > 20),
    )
    satellite_features = pd.DataFrame([features for features in satellite_features if features])

    # Drop rows where bounding box contained too many clouds
    if config.max_cloud_percent is not None:
        logger.info(
            f"Dropping {(satellite_features.cloud_pct > config.max_cloud_percent).sum():,} row(s) where bounding box has too many clouds."
        )
        satellite_features = satellite_features[
            satellite_features.cloud_pct <= config.max_cloud_percent
        ]

    # Drop rows where bounding box did not contain any water
    if config.filter_to_water_area:
        logger.info(
            f"Dropping {(satellite_features.num_water_pixels == 0).sum():,} row(s) where bouding box does not contain any water."
        )
        satellite_features = satellite_features[satellite_features.num_water_pixels > 0]

    # Drop where missing pixels (features will be nan)
    logger.info(
        f"Dropping {satellite_features.isna().any(axis=1).sum():,} row(s) where bouding box contains missing pixels."
    )
    satellite_features = satellite_features[~satellite_features.isna().any(axis=1)]

    # Winsorize top and bottom 1% of satellite band features
    satellite_features[config.satellite_image_features] = satellite_features[
        config.satellite_image_features
    ].apply(lambda x: winsorize(x, limits=(0.01, 0.01)))

    # Add in satellite meta features
    satellite_features = satellite_features.merge(
        satellite_meta[
            list(
                # use set to avoid duplication since days_before_sample may be in config.satellite_meta_features
                set(
                    ["item_id", "sample_id", "days_before_sample", "visual_href"]
                    + config.satellite_meta_features
                )
            )
        ],
        how="left",
        on=["item_id", "sample_id"],
        validate="1:1",
    )

    # Check that each row is a unique item / sample combo
    if satellite_features[["sample_id", "item_id"]].duplicated().any():
        raise ValueError(
            "There are repeat sample / item combinations in the satellite features dataframe"
        )

    # Use the most recent (and cloud-free if calculated) valid image
    cols_to_sort_on = ["days_before_sample"] + (
        ["cloud_pct"] if "cloud_pct" in satellite_features.columns else []
    )
    satellite_features = (
        satellite_features.sort_values(cols_to_sort_on).reset_index().groupby("sample_id").first()
    )

    return satellite_features


def _calculate_satellite_features_for_sample_item(
    sample_id: str, item_id: str, config: FeaturesConfig, cache_dir: Path
):
    """Generate the satellite features for specific combination of
    sample and pystac item ID
    """
    sample_item_dir = (
        Path(cache_dir) / f"sentinel_{config.image_feature_meter_window}/{sample_id}/{item_id}"
    )

    sample_item_features = {"sample_id": sample_id, "item_id": item_id}

    # Skip combos we were not able to download
    if not sample_item_dir.exists():
        return None

    # Load SCL band if filtering based on clouds and/or water
    if config.max_cloud_percent is not None or config.filter_to_water_area:
        scl_array = np.load(sample_item_dir / "SCL.npy")

    # Don't calculate features for the item if the cloud ratio is too high
    if config.max_cloud_percent is not None:
        cloud_ratio = ((scl_array >= 7) & (scl_array <= 10)).sum() / (
            scl_array.shape[1] * scl_array.shape[2]
        )
        sample_item_features["cloud_pct"] = cloud_ratio
        if cloud_ratio > config.max_cloud_percent:
            return sample_item_features

    # Load band arrays into a dictionary with band names for keys
    band_arrays = {}
    for band in config.use_sentinel_bands:
        if not (sample_item_dir / f"{band}.npy").exists():
            raise FileNotFoundError(
                f"Band {band} is missing from pystac item directory {sample_item_dir}"
            )
        arr = np.load(sample_item_dir / f"{band}.npy")

        # Set no data value to be nan
        arr = np.where(arr == 0, np.nan, arr)
        sample_item_features["num_no_data"] = np.isnan(arr).sum()

        # Filter array to water area
        if config.filter_to_water_area:
            if band != "SCL":
                scaled_scl = cv2.resize(scl_array[0], (arr.shape[2], arr.shape[1]))
                arr = arr[0][scaled_scl == 6]
                sample_item_features["num_water_pixels"] = arr.size

        # If the bounding box does not contain any water pixels (if filtering) or has entirely no data pixels, do not calculate features
        if arr.size == 0 or np.isnan(arr).all():
            return sample_item_features

        band_arrays[band] = arr

    # Iterate over features to generate
    for feature in config.satellite_image_features:
        # note: features will be nan if any pixel in bounding box is nan
        sample_item_features[feature] = SATELLITE_FEATURE_CALCULATORS[feature](band_arrays)

    return sample_item_features


def calculate_metadata_features(samples: pd.DataFrame, config: FeaturesConfig) -> pd.DataFrame:
    """Generate features from sample metadata

    Args:
        samples (pd.DataFrame): Dataframe where the index is sample_id and there are
            columns for date, longitude, and latitude
        config (FeaturesConfig): Feature configuration

    Returns:
        pd.DataFrame: Dataframe where the index is sample_id. There is one column
            for each metadata feature and one row for each sample
    """
    # Generate features for each sample
    sample_meta_features = samples.copy()

    # Pull in land cover classification from CDRP
    if "land_cover" in config.sample_meta_features:
        lc_cache_dir = Path(platformdirs.user_cache_dir()) / "cyfi"
        lc_cache_dir.mkdir(exist_ok=True)
        land_cover_map_filepath = lc_cache_dir / "C3S-LC-L4-LCCS-Map-300m-P1Y-2020-v2.1.1.nc"

        if not land_cover_map_filepath.exists():
            logger.debug(f"Downloading ~2GB land cover map to {lc_cache_dir}")
            s3p = S3Client(no_sign_request=True).S3Path(
                "s3://drivendata-public-assets/land_cover_map.tar.gz"
            )
            s3p.download_to(lc_cache_dir)
            file = tarfile.open(lc_cache_dir / "land_cover_map.tar.gz")
            file.extractall(lc_cache_dir)

        land_cover_data = xr.open_dataset(
            lc_cache_dir / "C3S-LC-L4-LCCS-Map-300m-P1Y-2020-v2.1.1.nc"
        )
        logger.info(
            f"Generating land cover features for {sample_meta_features.shape[0]:,} sample points."
        )
        land_covers = process_map(
            functools.partial(lookup_land_cover, land_cover_data=land_cover_data),
            sample_meta_features.latitude,
            sample_meta_features.longitude,
            chunksize=1,
            total=len(sample_meta_features),
            # Only log progress bar if debug message is logged
            disable=(logger._core.min_level > 20),
        )

        sample_meta_features["land_cover"] = land_covers

    if "rounded_longitude" in config.sample_meta_features:
        sample_meta_features["rounded_longitude"] = (sample_meta_features.longitude / 10).round(0)
    if "rounded_latitude" in config.sample_meta_features:
        sample_meta_features["rounded_latitude"] = (sample_meta_features.latitude / 10).round(0)

    return sample_meta_features[config.sample_meta_features]


def lookup_land_cover(latitude: float, longitude: float, land_cover_data: xr.Dataset) -> int:
    """Get the land cover classification value for a specific location

    Args:
        latitude (float): Latitude
        longitude (Longitude): Longitude
        land_cover_data (xr.Dataset): xarray Dataset with land cover information
    """
    return land_cover_data.sel(lat=latitude, lon=longitude, method="nearest").lccs_class.data[0]


def generate_all_features(
    samples: pd.DataFrame,
    satellite_meta: pd.DataFrame,
    config: FeaturesConfig,
    cache_dir: Union[str, Path],
) -> pd.DataFrame:
    """Generate a dataframe of features for the given set of samples.
    Requires that the raw satellite data for the given samples are
    already saved in cache_dir

    Args:
        samples (pd.DataFrame): Dataframe where the index is sample_id and there are
            columns for date, longitude, and latitude
        satellite_meta (pd.DataFrame): Dataframe of satellite metadata
            for all pystac items that have been selected for use in
            feature generation
        config (FeaturesConfig): Feature configuration
        cache_dir (Union[str, Path]): Cache directory where raw imagery
            is saved

    Returns:
        (selected_image_meta, features): Tuple of dataframes.

            'selected_image_meta' is a DataFrame with metadata and the
            visual_href link to the selected Sentinel image for each sample point.

            'features' is a dataframe where the index is sample_id and there is one
            column for each feature. Each row is a unique combination of
            sample and pystac item. Only samples that have at least one valid
            non-metadata feature are included in the features dataframe
    """
    # Generate satellite features, only includes samples with imagery
    satellite_features = calculate_satellite_features(satellite_meta, config, cache_dir)

    ct_with_satellite = satellite_features.index.nunique()
    if ct_with_satellite < samples.shape[0]:
        logger.warning(
            f"Relevant satellite data is not available for all sample points. Features will only be generated for {ct_with_satellite:,} sample points with valid satellite imagery ({(ct_with_satellite / samples.shape[0]):.0%} of sample points)"
        )

    features = satellite_features.drop(columns=["item_id"])

    # Generate non-satellite features. Each has only one row per sample
    if config.sample_meta_features:
        sample_meta_features = calculate_metadata_features(samples, config)
        # Don't include samples for which we only have metadata
        features = features.merge(
            sample_meta_features, left_index=True, right_index=True, how="left", validate="m:1"
        )

    all_feature_cols = (
        config.satellite_image_features
        + config.satellite_meta_features
        + config.sample_meta_features
    )
    features = features[all_feature_cols]
    ct_with_features = features.index.nunique()
    num_sat_features = len(config.satellite_image_features) + len(config.satellite_meta_features)
    if config.sample_meta_features:
        logger.info(
            f"Generated {num_sat_features:,} satellite feature(s) and {len(config.sample_meta_features):,} sample metadata feature(s) for {ct_with_features:,} sample points ({(ct_with_features / samples.shape[0]):.0%} of sample points)"
        )
    else:
        logger.info(
            f"Generated {num_sat_features:,} satellite feature(s) for {ct_with_features:,} sample points ({(ct_with_features / samples.shape[0]):.0%} of sample points)"
        )

    # Process string column values (eg replace `:` from satellite meta)
    features.columns = [col.replace(":", "_") for col in features.columns]

    # save out metadata on chosen satellite image for use in visualization later
    # only some of these are calculated depending on what filters are used (e.g. cloud and water)
    possible_meta_columns = [
        "item_id",
        "cloud_pct",
        "num_water_pixels",
        "days_before_sample",
        "visual_href",
    ]
    selected_image_meta = satellite_features[
        [c for c in possible_meta_columns if c in satellite_features.columns]
    ]
    return selected_image_meta, features
