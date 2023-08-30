## Code to generate features from raw downloaded source data
import functools
import os
import tarfile
from typing import Union

import appdirs
from cloudpathlib import AnyPath, S3Path
from loguru import logger
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.contrib.concurrent import process_map
import xarray as xr


from cyano.config import FeaturesConfig

# Create a dictionary mapping feature names to feature generator
# functions, which take a dictionary of band arrays as input
SATELLITE_FEATURE_CALCULATORS = {
    "NDVI_B04": lambda x: (x["B08"].mean() - x["B04"].mean())
    / (x["B08"].mean() + x["B04"].mean() + 1),
    "NDVI_B05": lambda x: (x["B08"].mean() - x["B05"].mean())
    / (x["B08"].mean() + x["B05"].mean() + 1),
    "NDVI_B06": lambda x: (x["B08"].mean() - x["B06"].mean())
    / (x["B08"].mean() + x["B06"].mean() + 1),
    "NDVI_B07": lambda x: (x["B08"].mean() - x["B07"].mean())
    / (x["B08"].mean() + x["B07"].mean() + 1),
    "green_red_ratio": lambda x: x["B03"].mean() / (x["B04"].mean() + 1),
    "green_blue_ratio": lambda x: x["B03"].mean() / (x["B02"].mean() + 1),
    "red_blue_ratio": lambda x: x["B04"].mean() / (x["B02"].mean() + 1),
    "green95th_blue_ratio": lambda x: np.percentile(x["B03"], 95) / (x["B02"].mean() + 1),
    "green5th_blue_ratio": lambda x: np.percentile(x["B03"], 5) / (x["B02"].mean() + 1),
    "prop_water": lambda x: (x["SCL"] == 6).mean(),
    "AOT_mean": lambda x: x["AOT"].mean(),
    "AOT_min": lambda x: x["AOT"].min(),
    "AOT_max": lambda x: x["AOT"].max(),
    "AOT_range": lambda x: x["AOT"].max() - x["AOT"].min(),
    "B01_mean": lambda x: x["B01"].mean(),
    "B01_min": lambda x: x["B01"].min(),
    "B01_max": lambda x: x["B01"].max(),
    "B01_range": lambda x: x["B01"].max() - x["B01"].min(),
    "B02_mean": lambda x: x["B02"].mean(),
    "B02_min": lambda x: x["B02"].min(),
    "B02_max": lambda x: x["B02"].max(),
    "B02_range": lambda x: x["B02"].max() - x["B02"].min(),
    "B03_mean": lambda x: x["B03"].mean(),
    "B03_min": lambda x: x["B03"].min(),
    "B03_max": lambda x: x["B03"].max(),
    "B03_range": lambda x: x["B03"].max() - x["B03"].min(),
    "B03_95th": lambda x: np.percentile(x["B03"], 95),
    "B03_5th": lambda x: np.percentile(x["B03"], 5),
    "B04_mean": lambda x: x["B04"].mean(),
    "B04_min": lambda x: x["B04"].min(),
    "B04_max": lambda x: x["B04"].max(),
    "B04_range": lambda x: x["B04"].max() - x["B04"].min(),
    "B05_mean": lambda x: x["B05"].mean(),
    "B05_min": lambda x: x["B05"].min(),
    "B05_max": lambda x: x["B05"].max(),
    "B05_range": lambda x: x["B05"].max() - x["B05"].min(),
    "B06_mean": lambda x: x["B06"].mean(),
    "B06_min": lambda x: x["B06"].min(),
    "B06_max": lambda x: x["B06"].max(),
    "B06_range": lambda x: x["B06"].max() - x["B06"].min(),
    "B07_mean": lambda x: x["B07"].mean(),
    "B07_min": lambda x: x["B07"].min(),
    "B07_max": lambda x: x["B07"].max(),
    "B07_range": lambda x: x["B07"].max() - x["B07"].min(),
    "B08_mean": lambda x: x["B08"].mean(),
    "B08_min": lambda x: x["B08"].min(),
    "B08_max": lambda x: x["B08"].max(),
    "B08_range": lambda x: x["B08"].max() - x["B08"].min(),
    "B09_mean": lambda x: x["B09"].mean(),
    "B09_min": lambda x: x["B09"].min(),
    "B09_max": lambda x: x["B09"].max(),
    "B09_range": lambda x: x["B09"].max() - x["B09"].min(),
    "B11_mean": lambda x: x["B11"].mean(),
    "B11_min": lambda x: x["B11"].min(),
    "B11_max": lambda x: x["B11"].max(),
    "B11_range": lambda x: x["B11"].max() - x["B11"].min(),
    "B12_mean": lambda x: x["B12"].mean(),
    "B12_min": lambda x: x["B12"].min(),
    "B12_max": lambda x: x["B12"].max(),
    "B12_range": lambda x: x["B12"].max() - x["B12"].min(),
    "B8A_mean": lambda x: x["B8A"].mean(),
    "B8A_min": lambda x: x["B8A"].min(),
    "B8A_max": lambda x: x["B8A"].max(),
    "B8A_range": lambda x: x["B8A"].max() - x["B8A"].min(),
    "SCL_mean": lambda x: x["SCL"].mean(),
    "SCL_min": lambda x: x["SCL"].min(),
    "SCL_max": lambda x: x["SCL"].max(),
    "SCL_range": lambda x: x["SCL"].max() - x["SCL"].min(),
    "WVP_mean": lambda x: x["WVP"].mean(),
    "WVP_min": lambda x: x["WVP"].min(),
    "WVP_max": lambda x: x["WVP"].max(),
    "WVP_range": lambda x: x["WVP"].max() - x["WVP"].min(),
}


def generate_features_for_sample_item(
    sample_id: str, item_id: str, config: FeaturesConfig, cache_dir: AnyPath
):
    """Generate the satellite features for specific combination of
    sample and pystac item ID
    """
    sample_item_dir = (
        Path(cache_dir) / f"sentinel_{config.image_feature_meter_window}/{sample_id}/{item_id}"
    )

    # Skip combos we were not able to download
    if not sample_item_dir.exists():
        return None

    # Load band arrays into a dictionary with band names for keys
    band_arrays = {}
    # If we want to mask image data with water boundaries in some way, add here
    for band in config.use_sentinel_bands:
        if not (sample_item_dir / f"{band}.npy").exists():
            raise FileNotFoundError(
                f"Band {band} is missing from pystac item directory {sample_item_dir}"
            )
        band_arrays[band] = np.load(sample_item_dir / f"{band}.npy")

    # Iterate over features to generate
    sample_item_features = {"sample_id": sample_id, "item_id": item_id}
    for feature in config.satellite_image_features:
        sample_item_features[feature] = SATELLITE_FEATURE_CALCULATORS[feature](band_arrays)

    return sample_item_features


def generate_satellite_features(
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
    # Determine the number of processes to use when parallelizing
    NUM_PROCESSES = int(os.getenv("CY_NUM_PROCESSES", 4))

    # Calculate satellite metadata features
    if "month" in config.satellite_meta_features:
        satellite_meta["month"] = pd.to_datetime(satellite_meta.datetime).dt.month

    logger.info(
        f"Generating satellite features for {len(satellite_meta):,} sample/item combos, {satellite_meta.sample_id.nunique():,} samples with {NUM_PROCESSES} processes"
    )

    # Iterate over selected sample / item combinations
    satellite_features = process_map(
        functools.partial(generate_features_for_sample_item, config=config, cache_dir=cache_dir),
        satellite_meta.sample_id,
        satellite_meta.item_id,
        chunksize=1,
        total=len(satellite_meta),
        max_workers=NUM_PROCESSES,
    )

    satellite_features = pd.DataFrame([features for features in satellite_features if features])

    # Add in satellite meta features
    satellite_features = satellite_features.merge(
        satellite_meta[["item_id", "sample_id"] + config.satellite_meta_features],
        how="left",
        on=["item_id", "sample_id"],
        validate="1:1",
    )

    # Check that each row is a unique item / sample combo
    if satellite_features[["sample_id", "item_id"]].duplicated().any():
        raise ValueError(
            "There are repeat sample / item combinations in the satellite features dataframe"
        )

    return satellite_features.set_index("sample_id").drop(columns=["item_id"]).astype(float)


def land_cover_for_sample(latitude: float, longitude: float, land_cover_data: xr.Dataset) -> int:
    """Get the land cover classification value for a specific location

    Args:
        latitude (float): Latitude
        longitude (Longitude): Longitude
        land_cover_data (xr.Dataset): xarray Dataset with land cover information
    """
    return land_cover_data.sel(lat=latitude, lon=longitude, method="nearest").lccs_class.data[0]


def generate_metadata_features(samples: pd.DataFrame, config: FeaturesConfig) -> pd.DataFrame:
    """Generate features from sample metadata

    Args:
        samples (pd.DataFrame): Dataframe where the index is sample_id and there are
            columns for date, longitude, and latitude
        config (FeaturesConfig): Feature configuration

    Returns:
        pd.DataFrame: Dataframe where the index is sample_id. There is one column
            for each metadata feature and one row for each sample
    """
    # Determine the number of processes to use when parallelizing
    NUM_PROCESSES = int(os.getenv("CY_NUM_PROCESSES", 4))

    # Generate features for each sample
    metadata_features = samples.copy()

    # Pull in land cover classification from CDRP
    if "land_cover" in config.metadata_features:
        lc_cache_dir = Path(appdirs.user_cache_dir()) / "cyano"
        lc_cache_dir.mkdir(exist_ok=True)
        land_cover_map_filepath = lc_cache_dir / "C3S-LC-L4-LCCS-Map-300m-P1Y-2020-v2.1.1.nc"

        if land_cover_map_filepath.exists():
            logger.debug(f"Using land cover map already downloaded to {lc_cache_dir}")
        else:
            logger.debug(f"Downloading ~2GB land cover map to {lc_cache_dir}")
            s3p = S3Path("s3://drivendata-public-assets/land_cover_map.tar.gz")
            s3p.download_to(lc_cache_dir)
            file = tarfile.open(lc_cache_dir / "land_cover_map.tar.gz")
            file.extractall(lc_cache_dir)

        logger.info(f"Loading land cover features with {NUM_PROCESSES} processes")
        land_cover_data = xr.open_dataset(
            lc_cache_dir / "C3S-LC-L4-LCCS-Map-300m-P1Y-2020-v2.1.1.nc"
        )
        land_covers = process_map(
            functools.partial(land_cover_for_sample, land_cover_data=land_cover_data),
            metadata_features.latitude,
            metadata_features.longitude,
            chunksize=1,
            total=len(metadata_features),
            max_workers=NUM_PROCESSES,
        )

        metadata_features["land_cover"] = land_covers

    if "rounded_longitude" in config.metadata_features:
        metadata_features["rounded_longitude"] = (metadata_features.longitude / 10).round(0)
    if "rounded_latitude" in config.metadata_features:
        metadata_features["rounded_latitude"] = (metadata_features.latitude / 10).round(0)

    return metadata_features[config.metadata_features]


def generate_features(
    samples: pd.DataFrame,
    satellite_meta: pd.DataFrame,
    config: FeaturesConfig,
    cache_dir: Union[str, Path],
) -> pd.DataFrame:
    """Generate a dataframe of features for the given set of samples.
    Requires that the raw satellite data for the give samples are
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
        pd.DataFrame: Dataframe where the index is sample_id and there is one
            column for each feature. Each row is a unique combination of
            sample and pystac item. Only samples that have at least one valid
            non-metadata feature are included in the features dataframe
    """
    # Generate satellite features
    # May be >1 row per sample, only includes samples with imagery
    satellite_features = generate_satellite_features(satellite_meta, config, cache_dir)
    logger.info(
        f"Generated {satellite_features.shape[1]} satellite features for {satellite_features.index.nunique():,} samples, {satellite_features.shape[0]:,} item/sample combinations."
    )

    # Generate non-satellite features. Each has only one row per sample
    features = satellite_features.copy()

    if config.metadata_features:
        metadata_features = generate_metadata_features(samples, config)
        logger.info(
            f"Generated {metadata_features.shape[1]} metadata features for {metadata_features.shape[0]:,} samples"
        )
        # Don't include samples for which we only have metadata
        features = features.merge(
            metadata_features, left_index=True, right_index=True, how="left", validate="m:1"
        )

    pct_with_features = features.index.nunique() / samples.shape[0]
    logger.success(
        f"Generated {features.shape[1]:,} features for {features.index.nunique():,} samples ({pct_with_features:.0%})"
    )

    all_feature_cols = (
        config.satellite_image_features + config.satellite_meta_features + config.metadata_features
    )

    return features[all_feature_cols]
