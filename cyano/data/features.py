## Code to generate features from raw downloaded source data
from typing import List, Union

from loguru import logger
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

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
    "blue_red_ratio": lambda x: x["B02"].mean() / x["B04"].mean(),
    "blue_green_ratio": lambda x: x["B02"].mean() / x["B03"].mean(),
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
        pd.DataFrame: Dataframe where the index is uid and there is one column
            for each satellite feature. There will only be rows for samples
            with satellite imagery
    """
    logger.info(
        f"Generating satellite features for {len(satellite_meta):,} sample/item combos, {satellite_meta.sample_id.nunique():,} samples"
    )
    # satellite_features_dict = {}
    satellite_features = []

    # Calculate satellite metadata features
    if "month" in config.satellite_meta_features:
        satellite_meta["month"] = pd.to_datetime(satellite_meta.datetime).dt.month

    # Iterate over selected sample / item combinations
    for row in tqdm(satellite_meta.itertuples(), total=len(satellite_meta)):
        sample_item_dir = (
            Path(cache_dir)
            / f"sentinel_{config.image_feature_meter_window}/{row.sample_id}/{row.item_id}"
        )
        # Skip combos we were not able to download
        if not sample_item_dir.exists():
            continue

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
        sample_item_features = {"sample_id": row.sample_id, "item_id": row.item_id}
        for feature in config.satellite_image_features:
            sample_item_features[feature] = SATELLITE_FEATURE_CALCULATORS[feature](band_arrays)
        satellite_features.append(pd.Series(sample_item_features))

    satellite_features = pd.concat(satellite_features, axis=1).T

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


def generate_climate_features(
    uids: Union[List[str], pd.Index], config: FeaturesConfig
) -> pd.DataFrame:
    """Generate features from climate data

    Args:
        uids (Union[List[str], pd.Index]): List of unique indices for each sample
        config (FeaturesConfig): Configuration, including
            directory where raw source data is saved

    Returns:
        pd.DataFrame: Dataframe where the index is uid. There is
            one columns for each climate feature and one row
            for each sample
    """
    # Load files
    # - filter to those containing '_climate' in the name or other pattern
    # - identify data for each sample based on uid

    # Generate features for each sample
    pass


def generate_elevation_features(
    uids: Union[List[str], pd.Index], config: FeaturesConfig
) -> pd.DataFrame:
    """Generate features from elevation data

    Args:
        uids (Union[List[str], pd.Index]): List of unique indices for each sample
        config (FeaturesConfig): Configuration, including
            directory where raw source data is saved

    Returns:
        pd.DataFrame: Dataframe where the index is uid. There is
            one columns for each elevation feature and one row
            for each sample
    """
    # Load files
    # - filter to those containing '_elevation' in the name or other pattern
    # - identify data for each sample based on uid

    # Generate features for each sample
    pass


def generate_metadata_features(samples: pd.DataFrame, config: FeaturesConfig) -> pd.DataFrame:
    """Generate features from sample metadata

    Args:
        samples (pd.DataFrame): Dataframe where the index is uid and there are
            columns for date, longitude, and latitude
        config (FeaturesConfig): Feature configuration

    Returns:
        pd.DataFrame: Dataframe where the index is uid. There is
            one columns for each metadata feature and one row
            for each sample
    """
    # Pull in any external information needed (eg land use by state)

    # Generate features for each sample
    metadata_features = samples.copy()
    if "rounded_longitude" in config.metadata_features:
        metadata_features["rounded_longitude"] = (metadata_features.longitude / 10).round(0)

    return metadata_features[config.metadata_features]


def generate_features(
    samples: pd.DataFrame,
    satellite_meta: pd.DataFrame,
    config: FeaturesConfig,
    cache_dir: Union[str, Path],
) -> pd.DataFrame:
    """Generate a dataframe of features for the given set of samples.
    Requires that the raw satellite, climate, and elevation data for
    the given samples are already saved in cache_dir

    Args:
        samples (pd.DataFrame): Dataframe where the index is uid and there are
            columns for date, longitude, and latitude
        satellite_meta (pd.DataFrame): Dataframe of satellite metadata
            for all pystac items that have been selected for use in
            feature generation
        config (FeaturesConfig): Feature configuration
        cache_dir (Union[str, Path]): Cache directory where raw imagery
            is saved

    Returns:
        pd.DataFrame: Dataframe where the index is uid and there is one
            column for each feature. Each row is a unique combination of
            sample and pystac item. Only samples that have valid satellite
            imagery are included in the features
    """
    # Generate satellite features
    # May be >1 row per sample, only includes samples with imagery
    satellite_features = generate_satellite_features(satellite_meta, config, cache_dir)
    logger.info(
        f"Generated {satellite_features.shape[1]} satellite features. {satellite_features.index.nunique():,} samples, {satellite_features.shape[0]:,} item / sample combinations."
    )
    feature_uids = satellite_features.index

    # Generate non-satellite features. Each has only one row per sample
    # Only include samples for which we have satellite features
    unique_feature_uids = satellite_features.index.unique
    non_satellite_features = []
    if config.climate_features:
        climate_features = generate_climate_features(unique_feature_uids, config, cache_dir)
        non_satellite_features.append(climate_features.loc[feature_uids])
        logger.info(f"Generated {satellite_features.shape[0]} climate features")

    if config.elevation_features:
        elevation_features = generate_elevation_features(unique_feature_uids, config, cache_dir)
        non_satellite_features.append(elevation_features.loc[feature_uids])
        logger.info(f"Generated {satellite_features.shape[0]} elevation features")

    if config.metadata_features:
        metadata_features = generate_metadata_features(samples, config)
        non_satellite_features.append(metadata_features.loc[feature_uids])

        logger.info(f"Generated {satellite_features.shape[0]} metadata features")
    non_satellite_features = pd.concat(non_satellite_features, axis=1)

    # Merge satellite and non-satellite features
    features = pd.concat([satellite_features, non_satellite_features], axis=1)

    all_feature_cols = (
        config.satellite_image_features
        + config.satellite_meta_features
        + config.climate_features
        + config.elevation_features
        + config.metadata_features
    )

    return features[all_feature_cols]
