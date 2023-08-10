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
    "ndvi_b04": lambda x: (x["B08"].mean() - x["B04"].mean())
    / (x["B08"].mean() + x["B04"].mean() + 1),
    "ndvi_b05": lambda x: (x["B08"].mean() - x["B05"].mean())
    / (x["B08"].mean() + x["B05"].mean() + 1),
    "ndvi_b06": lambda x: (x["B08"].mean() - x["B06"].mean())
    / (x["B08"].mean() + x["B06"].mean() + 1),
    "blue_red_ratio": lambda x: x["B02"].mean() / x["B04"].mean(),
    "blue_green_ratio": lambda x: x["B02"].mean() / x["B03"].mean(),
    "B02_mean": lambda x: x["B02"].mean(),
    "B02_min": lambda x: x["B02"].min(),
    "B02_max": lambda x: x["B02"].max(),
    "B03_mean": lambda x: x["B03"].mean(),
    "B03_min": lambda x: x["B03"].min(),
    "B03_max": lambda x: x["B03"].max(),
    "B04_mean": lambda x: x["B04"].mean(),
    "B04_min": lambda x: x["B04"].min(),
    "B04_max": lambda x: x["B04"].max(),
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

    # For now, fill missing imagery values with the average over all samples
    logger.info(
        f"Filling missing satellite values for {satellite_features.isna().any(axis=1).sum()} samples"
    )
    for col in config.satellite_image_features:
        satellite_features[col] = satellite_features[col].fillna(satellite_features[col].mean())

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

    return satellite_features.set_index("sample_id").drop(columns=["item_id"])


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
            column for each feature
    """
    # Generate satellite features
    # May be >1 row per sample, only includes samples with imagery
    satellite_features = generate_satellite_features(satellite_meta, config, cache_dir)
    logger.info(
        f"Generated {satellite_features.shape[1]} satellite features. {satellite_features.index.nunique():,} samples, {satellite_features.shape[0]:,} item / sample combinations."
    )

    # Generate non-satellite features. Each has only one row per sample
    uids = samples.index
    non_satellite_features = []
    if config.climate_features:
        climate_features = generate_climate_features(uids, config, cache_dir)
        non_satellite_features.append(climate_features.loc[uids])
        logger.info(f"Generated {satellite_features.shape[0]} climate features")

    if config.elevation_features:
        elevation_features = generate_elevation_features(uids, config, cache_dir)
        non_satellite_features.append(elevation_features.loc[uids])
        logger.info(f"Generated {satellite_features.shape[0]} elevation features")

    if config.metadata_features:
        metadata_features = generate_metadata_features(samples, config)
        non_satellite_features.append(metadata_features.loc[uids])

        logger.info(f"Generated {satellite_features.shape[0]} metadata features")
    non_satellite_features = pd.concat(non_satellite_features, axis=1)

    # Merge satellite and non-satellite features
    features = non_satellite_features.merge(
        satellite_features, how="outer", left_index=True, right_index=True, validate="1:m"
    )
    if set(features.index.unique()) != set(samples.index.unique()):
        raise ValueError(
            "The list of samples in the features dataframe does not match the original list of samples"
        )

    all_feature_cols = (
        config.satellite_image_features
        + config.satellite_meta_features
        + config.climate_features
        + config.elevation_features
        + config.metadata_features
    )

    return features[all_feature_cols]
