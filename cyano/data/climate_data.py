import functools
import os
from typing import List, Union

from herbie import FastHerbie
from loguru import logger
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from tqdm.contrib.concurrent import process_map

from cyano.config import FeaturesConfig

TRY_GRID_BUFFERS = np.arange(start=0.01, stop=0.1, step=0.01)


def process_samples_for_hrrr(samples: pd.DataFrame) -> pd.DataFrame:
    ## Process samples
    samples["date"] = pd.to_datetime(samples.date)
    # Drop samples from before HRRR was available
    samples = samples[samples.date > "2014-09-30"].copy()
    # Convert longitude and latitude
    for col in ["longitude", "latitude"]:
        samples[col] = np.where(samples[col] < 0, samples[col] + 360, samples[col])
    samples = samples.sort_values(by="date")

    return samples


def get_location_grid(
    latitude: float, longitude: float, xarrays_df: pd.DataFrame, try_grid_buffers=TRY_GRID_BUFFERS
):
    for buffer in try_grid_buffers:
        minlon = longitude - buffer
        maxlon = longitude + buffer
        minlat = latitude - buffer
        maxlat = latitude + buffer
        location_xarrays = xarrays_df[
            (xarrays_df.longitude >= minlon)
            & (xarrays_df.longitude <= maxlon)
            & (xarrays_df.latitude >= minlat)
            & (xarrays_df.latitude <= maxlat)
        ]
        # Is there at least one per example date?
        if len(location_xarrays) >= xarrays_df.valid_time.nunique():
            location_df = location_xarrays[["x", "y"]].drop_duplicates()
            location_df["latitude"] = latitude
            location_df["longitude"] = longitude

            return location_df

    logger.warning(f"Could not find any grid mapping for location ({latitude}, {longitude})")
    return None


def query_grid_mapping(samples: pd.DataFrame, example_dates: List = None) -> pd.DataFrame:
    """Returns a dataframe with one row for each combination of sample ID and
    HRRR data grid location (x,y)

    Args:
        samples (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    # Determine the number of processes to use when parallelizing
    NUM_PROCESSES = int(os.getenv("CY_NUM_PROCESSES", 4))

    ## Initialize FastHerbie with example dates
    if example_dates is None:
        example_dates = pd.concat(
            [
                samples.groupby(samples.date.dt.year).head(1),
                samples.groupby(samples.date.dt.year).tail(1),
            ]
        ).date.tolist()
    FH = FastHerbie(example_dates, model="hrrr", fxx=[0], verbose=False)
    # Read into an xarray and convert to dataframe
    herbie_xarray = FH.xarray("TMP:2 m", remove_grib=False)
    xarrays_df = herbie_xarray.to_dataframe().reset_index()

    ## Iterate over unique locations
    locations = samples.reset_index(drop=True)[["latitude", "longitude"]].drop_duplicates()
    # For each location, get all possible xs and ys in the grid
    logger.info(f"Iterating over {locations.shape[0]:,} locations to get HRRR grid indices")
    results = process_map(
        functools.partial(get_location_grid, xarrays_df=xarrays_df),
        locations.latitude,
        locations.longitude,
        max_workers=NUM_PROCESSES,
        chunksize=1,
        total=len(locations),
    )
    # Concatenate list of all x/ys for each location
    location_grid_map = pd.concat([res for res in results if res is not None]).reset_index(
        drop=True
    )

    # Merge back to sample IDs based on lat and long
    sample_grid_map = pd.merge(
        location_grid_map,
        samples.reset_index(),
        how="left",
        on=["latitude", "longitude"],
        validate="m:m",
    ).set_index("sample_id")

    return sample_grid_map[["x", "y", "date"]]


def load_hrrr_grid(samples: pd.DataFrame, cache_dir):
    samples = process_samples_for_hrrr(samples)

    ## Get mapping of sample locations to grid indices in HRRR
    grid_save_path = cache_dir / "hrrr_sample_grid_mapping.csv"

    # Load past grid mapping results if exist
    if grid_save_path.exists():
        sample_grid_map = pd.read_csv(grid_save_path, index_col=0)
        logger.info(
            f"Loaded past HRRR grid mapping results for {sample_grid_map.index.nunique():,} samples"
        )
        missing_samples = samples[~samples.index.isin(sample_grid_map.index)]
        if len(missing_samples) == 0:
            # Do not need to query for any additional samples
            return sample_grid_map.loc[samples.index]

        logger.info(f"Generating grid for remaining {missing_samples.shape[0]:,} samples")
        herbie_example_dates = pd.concat(
            [
                samples.groupby(samples.date.dt.year).head(1),
                samples.groupby(samples.date.dt.year).tail(1),
            ]
        ).date.tolist()
        new_sample_grid_map = query_grid_mapping(
            missing_samples, example_dates=herbie_example_dates
        )
        sample_grid_map = pd.concat([sample_grid_map, new_sample_grid_map])

    # Otherwise generate grid for all samples
    else:
        logger.info(f"Generating HRRR grid for {samples.shape[0]:,} samples")
        sample_grid_map = query_grid_mapping(samples)

    logger.info(
        f"Saving updated grid mapping with {sample_grid_map.index.nunique():,} samples to {grid_save_path}"
    )
    sample_grid_map.to_csv(grid_save_path, index=True)

    return sample_grid_map[sample_grid_map.index.isin(samples.index)]


def download_sample_climate(
    sample_id: str,
    date: Union[str, pd.Timestamp],
    latitude: float,
    longitude: float,
    save_dir: Path,
):
    """Query one sample's climate data based on its date, latitude,
    and longitude, and download the result.

    Args:
        sample_id (str): ID of the sample for which the item will be
            used when generating features
        date (Union[str, pd.Timestamp]): Sample date
        latitude (float): Sample latitude
        longitude (float): Sample longitude
        save_dir (Path): Directory to save all raw source data
    """
    # Query HRRR data

    # Save out data for sample
    pass


def download_climate_data(sample_list: pd.DataFrame, config: FeaturesConfig, cache_dir):
    """Query NOAA's HRRR database for a list of samples, and save out
    the raw results.

    Args:
        sample_list (pd.Dataframe): Dataframe with columns for date,
            longitude, latitude, and sample_id
        config (FeaturesConfig): Configuration, including
            directory to save raw source data
    """
    logger.info(f"Querying climate data for {sample_list.shape[0]:,} samples")

    # Iterate over samples (parallelize later)
    for sample in tqdm(sample_list.itertuples()):
        download_sample_climate(
            sample.Index,
            sample.date,
            sample.latitude,
            sample.longitude,
            save_dir=cache_dir,
        )
