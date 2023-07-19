## Code to generate features from raw downloaded source data
from typing import Dict, List, Union

from loguru import logger
import pandas as pd
from pathlib import Path


def generate_satellite_features(
    uids: Union[List[str], pd.Index], config: Dict, satellite_meta: pd.DataFrame
) -> pd.DataFrame:
    """Generate features from satellite data

    Args:
        uids (Union[List[str], pd.Index]): List of unique indices for each sample
        config (Dict): Experiment configuration, including directory where raw
            source data is saved
        satellite_meta (pd.DataFrame): Dataframe of metadata for the pystac items
            that will be used when generating features, including a columnmapping
            each sample ID to the relevant pystac item(s)

    Returns:
        pd.DataFrame: Dataframe where the index is uid and there is
            one columns for each satellite feature
    """
    # Load files
    # - identify data for each sample based satellite meta

    # Process data
    # - filter based on geographic area
    # - filter based on water boundary

    # Generate features for each sample
    pass


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


def generate_features(
    df: pd.DataFrame, config: Dict, satellite_meta: pd.DataFrame
) -> pd.DataFrame:
    """Generate a dataframe of features for the given set of samples.
    Requires that the raw satellite, climate, and elevation data for
    the given samples are already saved in features_dir

    Args:
        df (pd.DataFrame): Dataframe where the index is uid and there are
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
    uids = df.index
    satellite_features = generate_satellite_features(uids, config, satellite_meta)
    climate_features = generate_climate_features(uids, config)
    elevation_features = generate_elevation_features(uids, config)
    metadata_features = generate_metadata_features(df)

    logstr = (
        f"Features generated: {satellite_features.shape[0]} satellite, "
        f"{climate_features.shape[0]} climate, "
        f"{elevation_features.shape[0]} elevation, "
        f"and {metadata_features.shape[0]} metadata."
    )
    logger.info(logstr)

    features = pd.concat(
        [
            satellite_features.loc[uids],
            climate_features.loc[uids],
            elevation_features.loc[uids],
            metadata_features.loc[uids],
        ],
        axis=1,
    )

    return features
