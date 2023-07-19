from typing import List, Union

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
) -> List[pc.Item]:
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
        List[pc.Item]: List of all pystac items returned
    """
    # Define time search string

    # Generating search bounding box

    # Search planetary computer
    pass


def select_items(
    date: Union[str, pd.Timestamp], latitude: float, longitude: float, items: List[pc.Item]
) -> List[pc.Item]:
    """Select which pystac items to include for each sample

    Args:
        date (Union[str, pd.Timestamp]): Sample date
        latitude (float): Sample latitude
        longitude (float): Sample longitude
        items (List[pc.Item]): List of all pystac items returned by the planerary computer

    Returns:
        List[pc.Item]: List of pystac items to use when generating features for the sample
    """
    # Filter to items containing the sample point

    # Sort the possible items and determine which to select
    pass


def save_item(item: pc.Item, save_dir: Path, sample_id: str):
    """Save a pystac item

    Args:
        item (pc.Item): Item to save
        save_dir (Path): Directory to save all raw source data
        sample_id (str): ID of the sample for which the item will be
            used when generating features
    """
    pass


def download_satellite_data(
    sample_list: pd.Dataframe,
    save_dir: Path,
    collections: List[str] = ["sentinel-2-l2a", "landsat-c2-l2"],
    days_search_window: int = 15,
    meters_search_window: int = 50000,
):
    """Query the planetary computer for satellite data for a set of sample,
    select which items to include for each sample, and save out the raw results
    for processing.

    Args:
        sample_list (pd.Dataframe): Dataframe where the index is uid and there are
            columns for date, longitude, and latitude
        save_dir (Path): Directory to save all raw source data
        collections (List[str], optional): Satellite imagery collections to search in
            the planetary computer. Defaults to ["sentinel-2-l2a", "landsat-c2-l2"].
        days_search_window (int, optional): Number of days before the sample to
            include in the search. Defaults to 15.
        meters_search_window (int, optional): Buffer in meters to add on each side of
            the sample location when searching for imagery. Defaults to 50000.
    """
    logger.info(f"Querying satellite data for {sample_list.shape[0]:,} samples")

    # Iterate over samples (parallelize later)
    for sample in tqdm(sample_list.itertuples()):
        # Search planetary computer
        all_returned_items = search_planetary_computer(
            sample.date,
            sample.latitude,
            sample.longitude,
            collections=collections,
            days_search_window=days_search_window,
            meters_search_window=meters_search_window,
        )

        # Select items from results
        selected_items = select_items(
            sample.date, sample.latitude, sample.longitude, all_returned_items
        )

        # Save items
        for item in selected_items:
            save_item(item, save_dir, sample.Index)
