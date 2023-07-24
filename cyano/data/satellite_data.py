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
import rioxarray
from tqdm import tqdm

# Establish a connection to the STAC API
catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace
)


def get_bounding_box(latitude: float, longitude: float, meters_window: int) -> List[float]:
    """
    Given a latitude, longitude, and buffer in meters, returns a bounding
    box around the point with the buffer on the left, right, top, and bottom.

    Returns a list of [minx, miny, maxx, maxy]
    """
    distance_search = distance.distance(meters=meters_window)

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
        ItemSearch: Search results
    """
    # Define search query parameters
    bbox = get_bounding_box(latitude, longitude, meters_search_window)
    date_range = get_date_range(date, days_search_window)

    # Search planetary computer
    search_results = catalog.search(collections=collections, bbox=bbox, datetime=date_range)

    return search_results


def get_items_metadata(
    search_results: ItemSearch, latitude: float, longitude: float, config: Dict
) -> pd.DataFrame:
    """Get item metadata for a list of pystac items, including all information
    needed to select items for feature generation as well as the hrefs to
    download all relevant bands.

    Args:
        search_results (ItemSearch): _description_
        latitude (float): _description_
        longitude (float): _description_
        config (Dict): _description_

    Returns:
        pd.DataFrame: _description_
    """
    # Get metadata from pystac items
    items_meta = []
    for item in search_results.item_collection():
        item_meta = {
            "item_id": item.id,
            "datetime": item.datetime.strftime("%Y-%m-%d"),
            "min_long": item.bbox[0],
            "max_long": item.bbox[2],
            "min_lat": item.bbox[1],
            "max_lat": item.bbox[3],
        }
        if "eo:cloud_cover" in item.properties:
            item_meta.update({"cloud_cover": item.properties["eo:cloud_cover"]})
        # Add links to download each band needed for features
        for band in config["use_sentinel_bands"]:
            item_meta.update({f"{band}_href": item.assets[band].href})
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

    Returns:
        List[str]: List of the pystac items IDs for the selected items
    """
    # Select least cloudy item
    # items_meta.datetime = pd.to_datetime(items_meta.datetime)
    # items_meta["time_diff"] = np.abs(items_meta.datetime - pd.to_datetime(date))
    # closest_time = items_meta.sort_values(by="time_diff").iloc[0].item_id
    least_cloudy = items_meta.sort_values(by="cloud_cover").iloc[0].item_id

    return [least_cloudy]
    # return set([closest_time, least_cloudy])


def download_band(href: str, save_path: Path, bounding_box):
    """Download the geotiff for one band of a given pystac item
    based on the band's href, if the file does not already exist

    Args:
        href (str): _description_
        save_path (Path): _description_
    """

    if save_path.exists():
        return

    # Get image data
    (minx, miny, maxx, maxy) = bounding_box
    image_array = (
        rioxarray.open_rasterio(pc.sign(href))
        .rio.clip_box(
            minx=minx,
            miny=miny,
            maxx=maxx,
            maxy=maxy,
            crs="EPSG:4326",
        )
        .to_numpy()
    )

    # Save as numpy array
    np.save(save_path, image_array)


def identify_satellite_data(samples: pd.DataFrame, config: Dict):
    """_summary_

    Args:
        samples (pd.DataFrame): _description_
        config (Dict): _description_

    Returns:
        pd.DataFrame: Each row is a unique combination of sample ID
            and pystac item id
    """
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
        sample_items_meta = get_items_metadata(
            search_results, sample.latitude, sample.longitude, config
        )
        if len(sample_items_meta) == 0:
            no_results += 1
            continue

        # Select items to use for features
        selected_ids = select_items(sample_items_meta, sample.date)
        sample_items_meta["selected"] = sample_items_meta.item_id.isin(selected_ids)
        sample_items_meta["sample_id"] = sample.Index
        satellite_meta.append(sample_items_meta)

    logger.info(f"{no_results} samples did not return any satellite imagery results")

    # Concatenate satellite meta for all samples
    return pd.concat(satellite_meta)


def download_satellite_data(satellite_meta: pd.DataFrame, samples: pd.DataFrame, config: Dict):
    """_summary_

    Args:
        satellite_meta (pd.DataFrame): Each row is unique combo of satellite item ID and sample ID
        samples (pd.DataFrame): _description_
        config (Dict): _description_

    Raises:
        ValueError: _description_
    """
    # Filter to images selected for feature generation
    selected = satellite_meta[satellite_meta.selected]

    # Iterate over all rows (item / sample combos)
    logger.info(
        f"Downloading bands {config['use_sentinel_bands']} for {selected.shape[0]:,} items"
    )
    for _, download_row in tqdm(selected.iterrows(), total=len(selected)):
        sample_row = samples.loc[download_row.sample_id]
        sample_dir = Path(config["features_dir"]) / f"satellite/{download_row.sample_id}"
        sample_dir.mkdir(exist_ok=True, parents=True)

        save_bbox = get_bounding_box(
            sample_row.latitude, sample_row.longitude, config["image_feature_meter_window"]
        )
        # Iterate over bands to be saved
        for band in config["use_sentinel_bands"]:
            band_save_path = sample_dir / f"{download_row.item_id}_{band}.npy"
            download_band(download_row[f"{band}_href"], band_save_path, save_bbox)

        pass
