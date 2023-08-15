import hashlib

from cloudpathlib import AnyPath
from loguru import logger
import pandas as pd


def add_unique_identifier(df: pd.DataFrame) -> pd.DataFrame:
    """Given a dataframe with the columns []"latitude", "longitude", "date"],
    create a unique identifier for each row and set as the index

    Args:
        df (pd.DataFrame): Dataframe

    Returns:
        pd.DataFrame: Dataframe with unique identifiers as the index
    """
    df = df.copy()
    uids = []

    # create UID based on lat/lon and date
    for row in df.itertuples():
        m = hashlib.md5()
        for s in (row.latitude, row.longitude, row.date):
            m.update(str(s).encode())
        uids.append(m.hexdigest())

    df["sample_id"] = uids
    return df.set_index("sample_id")


def water_distance_filter(samples, distance_m=1000) -> pd.DataFrame:
    """Filter a list of samples to samples within a certain distance of
    water based on previous search of Google Earth Engine

    Args:
        samples (pd.DataFrame): Dataframe of samples where the index is sample_id
        distance_m (int, optional): Include sample points within this distance to
            water, in meters. Defaults to 1000.

    Returns:
        pd.DataFrame: Filtered dataframe of samples
    """
    # Load data with distance to water info + info to add identifiers
    train = pd.read_csv(
        AnyPath(
            "s3://drivendata-competition-nasa-cyanobacteria/experiments/splits/competition/train.csv"
        )
    )
    train = add_unique_identifier(train)
    test = pd.read_csv(
        AnyPath(
            "s3://drivendata-competition-nasa-cyanobacteria/experiments/splits/competition/test.csv"
        )
    )
    test = add_unique_identifier(test)
    distance = pd.concat([train.distance_to_water_m, test.distance_to_water_m])

    # Filter samples
    mask = distance.loc[samples.index] <= distance_m
    logger.info(
        f"Filtering from {samples.shape[0]:,} samples to {mask.sum():,} within {distance_m} m of water"
    )

    return samples[mask]
