from typing import List, Union

from herbie import FastHerbie
from loguru import logger
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from cyano.config import FeaturesConfig

TRY_GRID_BUFFERS = np.arange(start=0.01, stop=0.1, step=0.01)


def generate_grid_mapping(
    samples: pd.DataFrame, try_grid_buffers: List[float] = TRY_GRID_BUFFERS
) -> pd.DataFrame:
    ## Process samples
    samples["date"] = pd.to_datetime(samples.date)
    # Drop samples from before HRRR was available
    samples = samples[samples.date > "2014-09-30"].copy()
    # Convert longitude and latitude
    for col in ["longitude", "latitude"]:
        samples[col] = np.where(samples[col] < 0, samples[col] + 360, samples[col])
    samples = samples.sort_values(by="date")

    ## Initialize FastHerbie with example dates
    example_dates = pd.concat(
        [
            samples.groupby(samples.date.dt.year).head(1),
            samples.groupby(samples.date.dt.year).tail(1),
        ]
    ).date.tolist()
    logger.info(f"gathered {len(example_dates)} example dates")
    FH = FastHerbie(example_dates, model="hrrr", fxx=[0])
    # Read into an xarray and convert to dataframe
    herbie_xarray = FH.xarray("TMP:2 m", remove_grib=False)
    xarrays_df = herbie_xarray.to_dataframe().reset_index()
    logger.info(f"xarray has shape {xarrays_df.shape}")

    ## Iterate over unique locations
    locations = samples.reset_index(drop=True)[["latitude", "longitude"]].drop_duplicates()
    # For each location, get all possible xs and ys in the grid
    location_grid_maps = []
    for row in tqdm(locations.itertuples(), total=len(locations)):
        print(row.Index)
        for buffer in try_grid_buffers:
            print("\tbuffer:", buffer)
            minlon = row.longitude - buffer
            maxlon = row.longitude + buffer
            minlat = row.latitude - buffer
            maxlat = row.latitude + buffer
            location_xarrays = xarrays_df[
                (xarrays_df.longitude >= minlon)
                & (xarrays_df.longitude <= maxlon)
                & (xarrays_df.latitude >= minlat)
                & (xarrays_df.latitude <= maxlat)
            ]
            # Is there at least one per example date?
            if len(location_xarrays) >= len(example_dates):
                print("\tDone!")
                location_df = location_xarrays[["x", "y"]].drop_duplicates()
                location_df["latitude"] = row.latitude
                location_df["longitude"] = row.longitude
                location_grid_maps.append(location_df)
                break

    # Concatenate list of all x/ys for each location
    location_grid_map = pd.concat(location_grid_maps).reset_index(drop=True)

    # Merge back to sample IDs based on lat and long
    sample_grid_map = pd.merge(
        location_grid_map,
        samples.reset_index(),
        how="left",
        on=["latitude", "longitude"],
        validate="m:m",
    ).set_index("sample_id")

    return sample_grid_map[["x", "y", "date"]]


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
