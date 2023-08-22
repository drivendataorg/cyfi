import functools
import json
import os

from cloudpathlib import AnyPath
from loguru import logger
import pandas as pd
import planetary_computer
import pystac_client
import rioxarray
from tqdm.contrib.concurrent import process_map

from cyano.config import FeaturesConfig
from cyano.data.utils import get_bounding_box

# Establish a connection to the STAC API
catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)


def path_to_elevation_data(
    sample_id: str, elevation_feature_meter_window: int, cache_dir: AnyPath
) -> AnyPath:
    """Get the path to an expected file of elevation data based on the sample ID,
    feature buffer size in meters, and cache directory
    """
    return AnyPath(f"{cache_dir}/elevation_{elevation_feature_meter_window}/{sample_id}.json")


def download_sample_elevation(
    sample_id: str, longitude: float, latitude: float, meters_window: int, cache_dir: AnyPath
):
    """Query one sample's elevation data based on its latitude and longitude,
    and download the result. Save all possible stats that we may want to use
    during feature generation

    Args:
        sample_id (str): ID of the sample for which the item will be
            used when generating features
        latitude (float): Sample latitude
        longitude (float): Sample longitude
        meters_window (int): Buffer in meters around the point on each side
            to include in feature generation
        cache_dir (AnyPath): Cache directory
    """
    save_path = path_to_elevation_data(sample_id, meters_window, cache_dir)
    if save_path.exists():
        return

    search = catalog.search(
        collections=["cop-dem-glo-30"],
        intersects={"type": "Point", "coordinates": [longitude, latitude]},
    )
    items = list(search.items())
    if len(items) == 0:
        return

    signed_asset = planetary_computer.sign(items[0].assets["data"])
    ele_array = rioxarray.open_rasterio(signed_asset.href)

    bbox = get_bounding_box(latitude, longitude, meters_window=meters_window)
    cropped_ele_array = ele_array.rio.clip_box(
        minx=bbox[0], miny=bbox[1], maxx=bbox[2], maxy=bbox[3]
    )
    max_elev = cropped_ele_array.values.max()
    min_elev = cropped_ele_array.values.min()

    sample_features = {
        "elevation_at_sample": float(
            ele_array.sel(x=longitude, y=latitude, method="nearest").values[0]
        ),
        "elevation_max": float(max_elev),
        "elevation_min": float(min_elev),
        "elevation_range": float(max_elev - min_elev),
        "elevation_mean": float(cropped_ele_array.values.mean()),
        "elevation_std": float(cropped_ele_array.values.std()),
    }

    with open(save_path, "w") as fp:
        json.dump(sample_features, fp)


def download_elevation_data(samples: pd.DataFrame, config: FeaturesConfig, cache_dir: AnyPath):
    """Query Copernicus' DEM elevation database for a list of samples, and
    save out the raw results.

    Args:
        samples (pd.Dataframe): Dataframe with columns for date,
            longitude, latitude, and sample_id
        config (FeaturesConfig): Configuration, including
            directory to save raw source data
    """
    # Determine the number of processes to use when parallelizing
    NUM_PROCESSES = int(os.getenv("CY_NUM_PROCESSES", 4))

    logger.info(f"Querying elevation data for {samples.shape[0]:,} samples")

    # Create elevation directory
    (cache_dir / f"elevation_{config.elevation_feature_meter_window}").mkdir(
        exist_ok=True, parents=True
    )

    # Iterate over samples
    _ = process_map(
        functools.partial(
            download_sample_elevation,
            meters_window=config.elevation_feature_meter_window,
            cache_dir=cache_dir,
        ),
        samples.index,
        samples.latitude,
        samples.longitude,
        chunksize=1,
        total=len(samples),
        max_workers=NUM_PROCESSES,
    )
