## Code to generate features from raw downloaded source data
from typing import Dict, List, Union

from loguru import logger
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def generate_satellite_features(uids: Union[List[str], pd.Index], config: Dict) -> pd.DataFrame:
    """Generate features from satellite data

    Args:
        uids (Union[List[str], pd.Index]): List of unique indices for each sample
        config (Dict): Experiment configuration, including directory where raw
            source data is saved
        satellite_meta (pd.DataFrame): Dataframe of metadata for the pystac items
            that will be used when generating features, including a column mapping
            each sample ID to the relevant pystac item(s)

    Returns:
        pd.DataFrame: Dataframe where the index is uid and there is
            one columns for each satellite feature
    """
    logger.info(f"Generating features for {len(uids):,} samples")
    satellite_features_dict = {}
    for uid in tqdm(uids):
        satellite_features_dict[uid] = {}
        sample_dir = Path(config["features_dir"]) / f"satellite/{uid}"
        if not sample_dir.exists():
            continue

        # Generate features - min, mean, and max for selected bands
        # Right now we only have one item per sample, process will need to
        # change if we have multiple
        # For now based on fixed code, later make this more easily modular
        for band in config["use_sentinel_bands"]:
            band_path = list(sample_dir.glob(f"*{band}*.npy"))[0]
            if band_path.exists():
                image_arr = np.load(band_path)

                satellite_features_dict[uid][f"{band}_mean"] = image_arr.mean()
                satellite_features_dict[uid][f"{band}_min"] = image_arr.min()
                satellite_features_dict[uid][f"{band}_max"] = image_arr.max()

    satellite_features = pd.DataFrame(satellite_features_dict).T[config["satellite_features"]]

    # For now, fill missing values with the average over all samples
    logger.info(
        f"Filling {satellite_features.isna().sum().sum()} missing satellite values (across {satellite_features.isna().any(axis=1).sum()} samples)"
    )
    for col in satellite_features:
        satellite_features[col] = satellite_features[col].fillna(satellite_features[col].mean())

    return satellite_features

    # Load files
    # - identify data for each sample based satellite meta

    # Process data
    # - filter based on geographic area
    # - filter based on water boundary

    # Generate features for each sample


def generate_climate_features(uids: Union[List[str], pd.Index], config: Dict) -> pd.DataFrame:
    """Generate features from climate data

    Args:
        uids (Union[List[str], pd.Index]): List of unique indices for each sample
        config (Dict): Experiment configuration, including directory where raw
            source data is saved

    Returns:
        pd.DataFrame: Dataframe where the index is uid and there is
            one columns for each climate feature
    """
    # Load files
    # - filter to those containing '_climate' in the name or other pattern
    # - identify data for each sample based on uid

    # Generate features for each sample
    pass


def generate_elevation_features(uids: Union[List[str], pd.Index], config: Dict) -> pd.DataFrame:
    """Generate features from elevation data

    Args:
        uids (Union[List[str], pd.Index]): List of unique indices for each sample
        config (Dict): Experiment configuration, including directory where raw
            source data is saved

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


def generate_features(samples: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Generate a dataframe of features for the given set of samples.
    Requires that the raw satellite, climate, and elevation data for
    the given samples are already saved in features_dir

    Args:
        samples (pd.DataFrame): Dataframe where the index is uid and there are
            columns for date, longitude, and latitude
        config (Dict): Experiment configuration, including directory where raw
            source data is saved
        satellite_meta (pd.DataFrame): Dataframe of metadata for the pystac items
            that will be used when generating features, including a columnmapping
            each sample ID to the relevant pystac item(s)

    Returns:
        pd.DataFrame: Dataframe where the index is uid and there is one
            column for each feature
    """
    uids = samples.index
    all_features = []
    satellite_features = generate_satellite_features(uids, config)
    all_features.append(satellite_features.loc[uids])
    logger.info(f"Generated {satellite_features.shape[0]} satellite features")
    if config["climate_features"]:
        climate_features = generate_climate_features(uids, config)
        all_features.append(climate_features.loc[uids])
        logger.info(f"Generated {satellite_features.shape[0]} climate features")
    if config["elevation_features"]:
        elevation_features = generate_elevation_features(uids, config)
        all_features.append(elevation_features.loc[uids])
        logger.info(f"Generated {satellite_features.shape[0]} elevation features")
    if config["metadata_features"]:
        metadata_features = generate_metadata_features(samples)
        all_features.append(metadata_features.loc[uids])
        logger.info(f"Generated {satellite_features.shape[0]} metadata features")

    features = pd.concat(
        all_features,
        axis=1,
    )

    return features
