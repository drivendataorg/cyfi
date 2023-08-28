import os

from datetime import timedelta
import functools
import json
import shutil
from typing import Dict, List, Tuple, Union

from cloudpathlib import AnyPath
import geopy.distance as distance
from loguru import logger
import numpy as np
import pandas as pd
from pathlib import Path
import planetary_computer as pc
from pystac_client import Client, ItemSearch
import rioxarray
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from cyano.config import FeaturesConfig

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


def get_date_range(date: str, days_window: int) -> str:
    """Get a date range to search for in the planetary computer based
    on a sample's date. The time range will go from time_buffer_days
    before the sample date to the sample date

    Returns a string"""
    datetime_format = "%Y-%m-%d"
    date = pd.to_datetime(date)
    range_start = date - timedelta(days=days_window)
    date_range = f"{range_start.strftime(datetime_format)}/{date.strftime(datetime_format)}"

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


def bbox_from_geometry(geometry: Dict) -> Dict:
    """For pystac items that don't have the bbox attribute, get the
    bbox from the geometry

    Args:
        geometry (Dict): A dictionary of geometry from item.geometry

    Returns:
        Dict: Dictionary with keys for min_long, max_long, min_lat,
            and max_lat
    """
    lons = [coord_pair[0] for coord_pair in geometry["coordinates"][0]]
    lats = [coord_pair[1] for coord_pair in geometry["coordinates"][0]]

    return {
        "min_long": min(lons),
        "max_long": max(lons),
        "min_lat": min(lats),
        "max_lat": max(lats),
    }


def get_items_metadata(
    search_results: ItemSearch,
    latitude: float,
    longitude: float,
    config: FeaturesConfig,
) -> pd.DataFrame:
    """Get item metadata for a list of pystac items returned for a given sample,
    including all information needed to select items for feature generation as
    well as the hrefs to download all relevant bands.

    Args:
        search_results (ItemSearch): Result of searching the planetary computer
            for the given item
        latitude (float): Sample latitude
        longitude (float): Sample longitude
        config (FeaturesConfig): Experiment configuration

    Returns:
        pd.DataFrame: Item metadata
    """
    # Get metadata from pystac items
    items_meta = []
    for item in search_results.item_collection():
        item_meta = {
            "item_id": item.id,
            "datetime": item.datetime.strftime("%Y-%m-%d"),
            "platform": item.properties["platform"],
        }
        # Add item bounding box
        if "bbox" in item.to_dict():
            item_meta.update(
                {
                    "min_long": item.bbox[0],
                    "max_long": item.bbox[2],
                    "min_lat": item.bbox[1],
                    "max_lat": item.bbox[3],
                }
            )
        elif "geometry" in item.to_dict():
            bbox_dict = bbox_from_geometry(item.geometry)
            item_meta.update(bbox_dict)

        for optional_item_property in ["eo:cloud_cover", "s2:nodata_pixel_percentage"]:
            if optional_item_property in item.properties:
                item_meta.update({optional_item_property: item.properties[optional_item_property]})
        # Add links to download each band needed for features
        for band in config.use_sentinel_bands:
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


def generate_candidate_metadata(
    samples: pd.DataFrame, config: FeaturesConfig
) -> Tuple[pd.DataFrame, Dict]:
    """Generate metadata for all of the satellite item candidates
    that could be used to generate features for each sample

    Args:
        samples (pd.DataFrame): Dataframe where the index is sample ID and
            there are columns for date, longitude, and latitude
        config (FeaturesConfig): Features config

    Returns:
        Tuple[pd.DataFrame, Dict]: Tuple of (metadata for all sentinel item
            candidates, dictionary mapping sample IDs to the relevant
            pystac item IDs)
    """
    logger.info("Generating metadata for all satellite item candidates")

    if len(samples) > 20:
        # Load from saved directory with search results for all competition data
        # Remove for final package
        pc_results_dir = (
            AnyPath("s3://drivendata-competition-nasa-cyanobacteria")
            / "data/interim/full_pc_search"
        )
        sentinel_meta = pd.read_csv(pc_results_dir / "sentinel_metadata.csv")
        logger.info(
            f"Loaded {sentinel_meta.shape[0]:,} rows of Sentinel candidate metadata from {pc_results_dir}"
        )
        with open(pc_results_dir / "sample_item_map.json", "r") as fp:
            sample_item_map = json.load(fp)

        return (sentinel_meta, sample_item_map)

    # Otherwise, search the planetary computer
    logger.info(
        f"Searching {config.pc_collections} within {config.pc_days_search_window} days and {config.pc_meters_search_window} meters"
    )
    sentinel_meta = []
    sample_item_map = {}
    for sample in tqdm(samples.itertuples(), total=len(samples)):
        # Search planetary computer
        search_results = search_planetary_computer(
            sample.date,
            sample.latitude,
            sample.longitude,
            collections=config.pc_collections,
            days_search_window=config.pc_days_search_window,
            meters_search_window=config.pc_meters_search_window,
        )

        # Get satelite metadata
        sample_items_meta = get_items_metadata(
            search_results, sample.latitude, sample.longitude, config
        )

        sample_item_map[sample.Index] = {
            "sentinel_item_ids": sample_items_meta.item_id.tolist()
            if len(sample_items_meta) > 0
            else []
        }
        sentinel_meta.append(sample_items_meta)
    sentinel_meta = (
        pd.concat(sentinel_meta).groupby("item_id", as_index=False).first().reset_index(drop=True)
    )
    logger.info(f"Generated metadata for {sentinel_meta.shape[0]:,} Sentinel item candidates")

    return (sentinel_meta, sample_item_map)


def select_items(
    items_meta: pd.DataFrame,
    date: Union[str, pd.Timestamp],
    config: FeaturesConfig,
) -> List[str]:
    """Select which pystac items to include from a dataframe of items
    based on a sample's date

    Args:
        item_meta (pd.DataFrame): Dataframe with metadata about all possible
            pystac items to include for the given sample
        date (Union[str, pd.Timestamp]): Date the sample was collected
        config (FeaturesConfig): Features config

    Returns:
        List[str]: List of the pystac items IDs for the selected items
    """
    # Calculate days between sample and image
    items_meta["days_before_sample"] = (
        pd.to_datetime(date) - pd.to_datetime(items_meta.datetime)
    ).dt.days
    # Filter by time frame
    items_meta = items_meta[
        items_meta.days_before_sample.between(0, config.pc_days_search_window)
    ].copy()

    # Sort and select
    selected = items_meta.sort_values(
        by=["eo:cloud_cover", "days_before_sample"], ascending=[True, True]
    ).head(config.n_sentinel_items)

    return selected.item_id.tolist()


def select_sample_items(
    sample_id: str,
    sample_date: Union[str, pd.Timestamp],
    sample_item_map: pd.DataFrame,
    candidate_sentinel_meta: pd.DataFrame,
    config: FeaturesConfig,
):
    sample_item_ids = sample_item_map[sample_id]["sentinel_item_ids"]
    if len(sample_item_ids) == 0:
        return None

    sample_items_meta = candidate_sentinel_meta[
        candidate_sentinel_meta.item_id.isin(sample_item_ids)
    ].copy()
    selected_ids = select_items(sample_items_meta, sample_date, config)

    sample_items_meta = sample_items_meta[sample_items_meta.item_id.isin(selected_ids)]
    sample_items_meta["sample_id"] = sample_id

    return sample_items_meta


def identify_satellite_data(samples: pd.DataFrame, config: FeaturesConfig) -> pd.DataFrame:
    """Identify all pystac items to be used during feature
    generation for a given set of samples

    Args:
        samples (pd.DataFrame): Dataframe where the index is sample_id and
            there are columns for date, longitude, and latitude
        config (FeaturesConfig): Features config

    Returns:
        pd.DataFrame: Each row is a unique combination of sample ID
            and pystac item id. The 'selected' column indicates
            which will be used in feature generation
    """
    # Determine the number of processes to use when parallelizing
    NUM_PROCESSES = int(os.getenv("CY_NUM_PROCESSES", 4))

    ## Get all candidate item metadata
    candidate_sentinel_meta, sample_item_map = generate_candidate_metadata(samples, config)

    ## Select which items to use for each sample
    logger.info(
        f"Selecting which satellite items to use for feature generation with {NUM_PROCESSES} processes"
    )

    selected_satellite_meta = process_map(
        functools.partial(
            select_sample_items,
            sample_item_map=sample_item_map,
            candidate_sentinel_meta=candidate_sentinel_meta,
            config=config,
        ),
        samples.index,
        samples.date,
        max_workers=NUM_PROCESSES,
        chunksize=1,
        total=len(samples),
    )

    # Drop values of None
    selected_satellite_meta = [items_meta for items_meta in selected_satellite_meta if items_meta]
    # Concatenate all satellite metadata
    selected_satellite_meta = pd.concat(selected_satellite_meta).reset_index(drop=True)

    # selected_satellite_meta = []
    # for sample in tqdm(samples.itertuples(), total=len(samples)):
    #     sample_item_ids = sample_item_map[sample.Index]["sentinel_item_ids"]
    #     if len(sample_item_ids) == 0:
    #         continue

    #     sample_items_meta = candidate_sentinel_meta[
    #         candidate_sentinel_meta.item_id.isin(sample_item_ids)
    #     ].copy()
    #     selected_ids = select_items(sample_items_meta, sample.date, config)

    #     # Save out the selected items
    #     sample_items_meta = sample_items_meta[sample_items_meta.item_id.isin(selected_ids)]
    #     sample_items_meta["sample_id"] = sample.Index

    #     selected_satellite_meta.append(sample_items_meta)

    # selected_satellite_meta = pd.concat(selected_satellite_meta).reset_index(drop=True)
    logger.info(
        f"Identified satellite imagery for {selected_satellite_meta.sample_id.nunique():,} samples"
    )

    return selected_satellite_meta


def download_row(
    iterrow: Tuple[int, pd.Series],
    samples: pd.DataFrame,
    imagery_dir: Path,
    config: FeaturesConfig,
):
    """Download image arrays for one row of satellite metadata containing a
    unique combination of sample ID and item ID

    Args:
        iterrow (Tuple[int, pd.Series]): One item in the generator produced
            by satellite_meta.iterrows(), where satellite_meta is a dataframe
            of satellite metadata for all pystac items that have been
            selected for use in feature generation
        samples (pd.DataFrame): Dataframe where the index is sample_id and
            there are columns for date, longitude, and latitude
        imagery_dir (Path): Image cache directory for a specific satellite
            source and bounding box size
        config (FeaturesConfig): Features config
    """
    _, row = iterrow

    sample_row = samples.loc[row.sample_id]
    sample_image_dir = imagery_dir / f"{row.sample_id}/{row.item_id}"
    sample_image_dir.mkdir(exist_ok=True, parents=True)

    # Get bounding box for array to save out
    (minx, miny, maxx, maxy) = get_bounding_box(
        sample_row.latitude,
        sample_row.longitude,
        config.image_feature_meter_window,
    )

    try:
        # Iterate over bands and save
        for band in config.use_sentinel_bands:
            # Check if the file already exists
            array_save_path = sample_image_dir / f"{band}.npy"
            if not array_save_path.exists():
                # Get unsigned URL so we don't use expired token
                unsigned_href = row[f"{band}_href"].split("?")[0]
                band_array = (
                    rioxarray.open_rasterio(pc.sign(unsigned_href))
                    .rio.clip_box(
                        minx=minx,
                        miny=miny,
                        maxx=maxx,
                        maxy=maxy,
                        crs="EPSG:4326",
                    )
                    .to_numpy()
                )
                np.save(array_save_path, band_array)

    except Exception as e:
        # Delete item directory if it has already been created
        if sample_image_dir.exists():
            shutil.rmtree(sample_image_dir)

        return f"{sample_image_dir.parts[-2]}/{sample_image_dir.parts[-1]}: {type(e)} {e}"


def download_satellite_data(
    satellite_meta: pd.DataFrame,
    samples: pd.DataFrame,
    config: FeaturesConfig,
    cache_dir: Union[str, Path],
):
    """Download satellite images as one stacked numpy arrays per pystac item

    Args:
        satellite_meta (pd.DataFrame): Dataframe of satellite metadata
            for all pystac items that have been selected for use in
            feature generation
        samples (pd.DataFrame): Dataframe where the index is sample_id and
            there are columns for date, longitude, and latitude
        config (FeaturesConfig): Features config
        cache_dir (Union[str, Path]): Cache directory to save raw imagery
    """
    # Determine the number of processes to use when parallelizing
    NUM_PROCESSES = int(os.getenv("CY_NUM_PROCESSES", 4))

    logger.info(f"Downloading bands {config.use_sentinel_bands} with {NUM_PROCESSES} processes")

    imagery_dir = Path(cache_dir) / f"sentinel_{config.image_feature_meter_window}"
    # Iterate over all rows (item / sample combos)
    exception_logs = process_map(
        functools.partial(
            download_row,
            samples=samples,
            imagery_dir=imagery_dir,
            config=config,
        ),
        satellite_meta.iterrows(),
        max_workers=NUM_PROCESSES,
        chunksize=1,
        total=len(satellite_meta),
    )
    exceptions = [e for e in exception_logs if e]
    if len(exceptions) > 0:
        # Log number of exceptions to CLI
        logger.warning(f"{len(exceptions):,} exceptions raised during download")
        # Log full list of exceptions to .log file
        exceptions = "\n".join(exceptions)
        logger.debug(f"Exceptions:\n{exceptions}")
