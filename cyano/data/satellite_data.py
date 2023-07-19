from typing import Dict, List, Union

from loguru import logger
import pandas as pd
from pathlib import Path
import planetary_computer as pc
from tqdm import tqdm


def search_planetary_computer(
    date: Union[str, pd.Timestamp],
    latitude: float,
    longitude: float,
    collections: List[str],
    days_search_window: int,
    meters_search_window: int,
) -> pd.DataFrame:
    """Search the planetary computer for imagery relevant to a given sample

    Args:
        date (Union[str, pd.Timestamp]): Sample date
        latitude (float): Sample latitude
        longitude (float): Sample longitude
        collections (List[str]): Satellite imagery collections to search
        days_search_window (int): Number of days before the sample to include
            in the search
        meters_search_window (int): Buffer in meters to add on each side of the
            sample location when searching for imagery

    Returns:
        pd.DataFrame: List of metadata about each pystac item for filtering,
            including a unique item identifier and a URL column that can
            be used to download the item
    """
    # Define time search string

    # Generating search bounding box

    # Search planetary computer

    # Generate dataframe of metadata
    pass


def select_items(
    item_meta: pd.DataFrame,
    date: Union[str, pd.Timestamp],
    latitude: float,
    longitude: float,
) -> pd.DataFrame:
    """Select which pystac items to include for a given sample

    Args:
        item_meta (pd.DataFrame): Dataframe with metadata about all possible
            pystac items to include for the given sample
        date (Union[str, pd.Timestamp]): Sample date
        latitude (float): Sample latitude
        longitude (float): Sample longitude

    Returns:
        pd.DataFrame: Dataframe of metadata for the pystac items that will be
            used when generating features for the sample
    """
    # Filter to items containing the sample point

    # Sort the possible items and determine which to select
    pass


def identify_satellite_data(samples, config):
    """Query the planetary computer for satellite data for a set of
    samples, and generate metadata for the selected pystac items
    for each sample
    """
    satellite_meta = []

    for sample in tqdm(samples.itertuples()):
        # Search planetary computer
        sample_item_meta = search_planetary_computer(
            sample.date,
            sample.latitude,
            sample.longitude,
            collections=config["collections"],
            days_search_window=config["days_search_window"],
            meters_search_window=config["meters_search_window"],
        )

        # Select items from results
        sample_item_meta = select_items(
            sample_item_meta, sample.date, sample.latitude, sample.longitude
        )

        sample_item_meta["sample_id"] = sample.Index
        satellite_meta.append(sample_item_meta)

    # Concatenate satellite meta for all samples
    satellite_meta = pd.concat(satellite_meta)

    return satellite_meta


def download_satellite_data(satellite_meta: pd.DataFrame, config: Dict):
    """Download the selected pystac items that will be used to generate features.
    Pystac items will be saved under an identifier unique to each item.
    satellite_meta maps which pystac item should be used to generate features for
    each sample

    Args:
        satellite_meta (pd.DataFrame): Dataframe of metadata for the pystac items
            that will be used when generating features
        config (Dict): Experiment configuration, including directory to save
            raw source data
    """
    # Get a list of the uniue different pystac items that need to be downloaded

    # Iterate over items and download
    pass
