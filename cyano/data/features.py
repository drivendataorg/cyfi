## Code to generate features from raw downloaded source data
from typing import List, Union

from loguru import logger
import pandas as pd
from pathlib import Path


def generate_climate_features(
    uids: Union[List[str], pd.Index], features_dir: Path
) -> pd.DataFrame:
    """Generate features from climate data

    Args:
        uids (List[str]): List of unique indices for each sample
        features_dir (Path): Directory with raw climate data

    Returns:
        pd.DataFrame: Dataframe where the index is uid and there is
         one columns for each climate feature
    """
    # Load files
    # - filter to those containing '_climate' in the name or other pattern
    # - identify data for each sample based on uid

    # Generate features
    pass


def generate_elevation_features(
    uids: Union[List[str], pd.Index], features_dir: Path
) -> pd.DataFrame:
    """Generate features from elevation data

    Args:
        uids (List[str]): List of unique indices for each sample
        features_dir (Path): Directory with raw elevation data

    Returns:
        pd.DataFrame: Dataframe where the index is uid and there is
         one columns for each elevation feature
    """
    # Load files
    # - filter to those containing '_elevation' in the name or other pattern
    # - identify data for each sample based on uid

    # Generate features
    pass


def generate_satellite_features(
    uids: Union[List[str], pd.Index], features_dir: Path
) -> pd.DataFrame:
    """Generate features from satellite data

    Args:
        uids (List[str]): List of unique indices for each sample
        features_dir (Path): Directory with raw satellite data

    Returns:
        pd.DataFrame: Dataframe where the index is uid and there is
         one columns for each satellite feature
    """
    # Load files
    # - identify data for each sample based on uid

    # Process data
    # - filter based on geographic area
    # - filter based on water boundary

    # Generate features
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

    # Generate features
    pass


def generate_features(df: pd.DataFrame, features_dir: Path) -> pd.DataFrame:
    """Generate a dataframe of features for the given set of samples.
    Requires that the raw satellite, climate, and elevation data for
    the given samples is already saved in features_dir

    Args:
        df (pd.DataFrame): Dataframe where the index is uid and there are
            columns for date, longitude, and latitude
        features_dir (Path): Directory in which raw satellite, climate,
            and elevation data for the given samples is saved

    Returns:
        pd.DataFrame: Dataframe where the index is uid and there is one
            column for each feature
    """
    uids = df.index
    satellite_features = generate_satellite_features(uids, features_dir)
    climate_features = generate_climate_features(uids, features_dir)
    elevation_features = generate_elevation_features(uids, features_dir)
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
