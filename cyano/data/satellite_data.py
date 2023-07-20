from datetime import timedelta
from typing import Dict, List, Union

import geopy.distance as distance
from loguru import logger
import numpy as np
import pandas as pd
from pathlib import Path
import planetary_computer as pc
import pystac
from pystac_client import Client, ItemSearch
from tqdm import tqdm

# Establish a connection to the STAC API
catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace
)


def get_bounding_box(latitude: float, longitude: float, meters_search_window: int) -> List[float]:
    """
    Given a latitude, longitude, and buffer in meters, returns a bounding
    box around the point with the buffer on the left, right, top, and bottom.

    Returns a list of [minx, miny, maxx, maxy]
    """
    distance_search = distance.distance(meters=meters_search_window)

    # calculate the lat/long bounds based on ground distance
    # bearings are cardinal directions to move (south, west, north, and east)
    min_lat = distance_search.destination((latitude, longitude), bearing=180)[0]
    min_long = distance_search.destination((latitude, longitude), bearing=270)[1]
    max_lat = distance_search.destination((latitude, longitude), bearing=0)[0]
    max_long = distance_search.destination((latitude, longitude), bearing=90)[1]

    return [min_long, min_lat, max_long, max_lat]


def get_date_range(date: str, days_search_window: int) -> str:
    """Get a date range to search for in the planetary computer based
    on a sample's date. The time range will go from time_buffer_days
    before the sample date to time_buffer_days after the sample date

    Returns a string"""
    datetime_format = "%Y-%m-%d"
    range_start = pd.to_datetime(date) - timedelta(days=days_search_window)
    range_end = pd.to_datetime(date) + timedelta(days=days_search_window)
    date_range = f"{range_start.strftime(datetime_format)}/{range_end.strftime(datetime_format)}"

    return date_range


def search_planetary_computer(
    date: Union[str, pd.Timestamp],
    latitude: float,
    longitude: float,
    collections: List[str],
    days_search_window: int,
    meters_search_window: int,
) -> ItemSearch:
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
    # Define search query parameters
    bbox = get_bounding_box(latitude, longitude, meters_search_window)
    date_range = get_date_range(date, days_search_window)

    # Search planetary computer
    search_results = catalog.search(collections=collections, bbox=bbox, datetime=date_range)

    return search_results


def get_items_metadata(
    search_results: ItemSearch, latitude: float, longitude: float
) -> pd.DataFrame:
    # Get metadata from pystac items
    items_meta = []
    for item in search_results.item_collection():
        item_meta = {
            "id": item.id,
            "datetime": item.datetime.strftime("%Y-%m-%d"),
            "min_long": item.bbox[0],
            "max_long": item.bbox[2],
            "min_lat": item.bbox[1],
            "max_lat": item.bbox[3],
        }
        if "eo:cloud_cover" in item.properties:
            item_meta.update({"cloud_cover": item.properties["eo:cloud_cover"]})
        items_meta.append(item_meta)
    items_meta = pd.DataFrame(items_meta)
    if len(items_meta) == 0:
        return items_meta

    # Filter to items containing the sample point
    items_meta = items_meta[
        (items_meta.min_lat < latitude)
        & (items_meta.max_lat > latitude)
        & (items_meta.min_long < longitude)
        & (items_meta.max_long > longitude)
    ]

    return items_meta


def select_items(
    items_meta: pd.DataFrame,
    date: Union[str, pd.Timestamp],
) -> List[str]:
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
    # Select closest in time and least cloudy
    items_meta.datetime = pd.to_datetime(items_meta.datetime)
    items_meta["time_diff"] = np.abs(items_meta.datetime - pd.to_datetime(date))
    closest_time = items_meta.sort_values(by="time_diff").iloc[0].id
    least_cloudy = items_meta.sort_values(by="cloud_cover").iloc[0].id

    return set([closest_time, least_cloudy])


def download_item(item: pystac.Item, save_dir: Path):
    item_save_path = save_dir / item.id
    if not item_save_path.exists():
        item.save_object(dest_href=item_save_path)
    pass


def download_satellite_data(samples, config: Dict):
    save_dir = Path(config["features_dir"]) / "satellite"
    save_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Saving pystac items to {save_dir}")
    logger.info(
        f"Searching {config['pc_collections']} within {config['pc_days_search_window']} days and {config['pc_meters_search_window']} meters"
    )

    satellite_meta = []
    no_results = 0
    for sample in tqdm(samples.itertuples(), total=len(samples)):
        # Search planetary computer
        search_results = search_planetary_computer(
            sample.date,
            sample.latitude,
            sample.longitude,
            collections=config["pc_collections"],
            days_search_window=config["pc_days_search_window"],
            meters_search_window=config["pc_meters_search_window"],
        )

        # Get satelite metadata
        sample_items_meta = get_items_metadata(search_results, sample.latitude, sample.longitude)
        if len(sample_items_meta) == 0:
            no_results += 1
            continue

        # Select items to use for features
        selected_ids = select_items(sample_items_meta, sample.date)
        sample_items_meta["selected"] = sample_items_meta.id.isin(selected_ids)
        sample_items_meta["sample_id"] = sample.Index
        satellite_meta.append(sample_items_meta)

        download_items = [
            item for item in search_results.item_collection() if item.id in selected_ids
        ]
        for item in download_items:
            download_item(item, save_dir)

    logger.info(f"{no_results} samples did not return any results")

    # Concatenate satellite meta for all samples and save
    satellite_meta = pd.concat(satellite_meta)
    save_satellite_to = Path(config["features_dir"]) / "satellite_metadata.csv"
    satellite_meta.to_csv(save_satellite_to, index=False)
    logger.info(
        f"{satellite_meta.shape[0]:,} rows of satellite metadata saved to {save_satellite_to}"
    )
