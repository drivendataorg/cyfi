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
    uids: Union[List[str], pd.Index], config: FeaturesConfig, cache_dir
) -> pd.DataFrame:
    """Generate features from satellite data

    Args:
        uids (Union[List[str], pd.Index]): List of unique indices for each sample
        config (FeaturesConfig): Configuration, including
            directory where raw source data is saved

    Returns:
        pd.DataFrame: Dataframe where the index is uid and there is one column
            for each satellite feature
    """
    logger.info(f"Generating features for {len(uids):,} samples")
    satellite_features_dict = {}
    # Iterate over samples
    for uid in tqdm(uids):
        satellite_features_dict[uid] = {}
        sample_dir = Path(cache_dir) / f"sentinel_{config.image_feature_meter_window}/{uid}"
        # Skip samples with no imagery
        if not sample_dir.exists():
            continue

        # Load band arrays for each image
        # Right now we only have one item per sample, process will need to
        # change if we have multiple
        item_dirs = list(sample_dir.iterdir())
        if len(item_dirs) == 0:
            continue
        elif len(item_dirs) > 1:
            raise NotImplementedError(
                f"{uid} has multiple items, cannot process multiple items per sample"
            )

        item_dir = item_dirs[0]
        # Load band arrays into a dictionary with band names for keys
        band_arrays = {}
        # If we want to mask image data with water boundaries in some way, add here
        for band in config.use_sentinel_bands:
            if not (item_dir / f"{band}.npy").exists():
                raise FileNotFoundError(
                    f"Band {band} is missing from pystac item directory {item_dir}"
                )
            band_arrays[band] = np.load(item_dir / f"{band}.npy")

        # Iterate over features to generate
        for feature in config.satellite_features:
            satellite_features_dict[uid][feature] = SATELLITE_FEATURE_CALCULATORS[feature](
                band_arrays
            )

    satellite_features = pd.DataFrame(satellite_features_dict).T[config.satellite_features]

    # For now, fill missing values with the average over all samples
    logger.info(
        f"Filling missing satellite values for {satellite_features.isna().any(axis=1).sum()} samples"
    )
    for col in satellite_features:
        satellite_features[col] = satellite_features[col].fillna(satellite_features[col].mean())

    return satellite_features


def generate_climate_features(
    uids: Union[List[str], pd.Index], config: FeaturesConfig
) -> pd.DataFrame:
    """Generate features from climate data

    Args:
        uids (Union[List[str], pd.Index]): List of unique indices for each sample
        config (FeaturesConfig): Configuration, including
            directory where raw source data is saved

    Returns:
        pd.DataFrame: Dataframe where the index is uid and there is
            one columns for each climate feature
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
        pd.DataFrame: Dataframe where the index is uid and there is
            one columns for each elevation feature
    """
    # Load files
    # - filter to those containing '_elevation' in the name or other pattern
    # - identify data for each sample based on uid

    # Generate features for each sample
    pass


def generate_metadata_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate features from sample metadata

    Args:
        df (pd.DataFrame): Dataframe where the index is uid and there are
            columns for date, longitude, and latitude

    Returns:
        pd.DataFrame: Dataframe where the index is uid and there is
            one columns for each metadata-based feature
    """
    # Pull in any external information needed (eg land use by state)

    # Generate features for each sample
    pass


def generate_features(samples: pd.DataFrame, config: FeaturesConfig, cache_dir) -> pd.DataFrame:
    """Generate a dataframe of features for the given set of samples.
    Requires that the raw satellite, climate, and elevation data for
    the given samples are already saved in cache_dir

    Args:
        samples (pd.DataFrame): Dataframe where the index is uid and there are
            columns for date, longitude, and latitude
        config (FeaturesConfig): Configuration, including
            directory where raw source data is saved

    Returns:
        pd.DataFrame: Dataframe where the index is uid and there is one
            column for each feature
    """
    uids = samples.index
    all_features = []
    satellite_features = generate_satellite_features(uids, config, cache_dir)
    all_features.append(satellite_features.loc[uids])
    logger.info(f"Generated {satellite_features.shape[1]} satellite features")

    if config.climate_features:
        climate_features = generate_climate_features(uids, config, cache_dir)
        all_features.append(climate_features.loc[uids])
        logger.info(f"Generated {satellite_features.shape[0]} climate features")

    if config.elevation_features:
        elevation_features = generate_elevation_features(uids, config, cache_dir)
        all_features.append(elevation_features.loc[uids])
        logger.info(f"Generated {satellite_features.shape[0]} elevation features")

    if config.metadata_features:
        metadata_features = generate_metadata_features(samples, cache_dir)
        all_features.append(metadata_features.loc[uids])
        logger.info(f"Generated {satellite_features.shape[0]} metadata features")

    features = pd.concat(
        all_features,
        axis=1,
    )

    return features
