from typing import Union

from loguru import logger
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def download_sample_elevation(
    sample_id: str,
    date: Union[str, pd.Timestamp],
    latitude: float,
    longitude: float,
    save_dir: Path,
):
    """Query one sample's elevation data based on its date, latitude,
    and longitude, and download the result.

    Args:
        sample_id (str): ID of the sample for which the item will be
            used when generating features
        date (Union[str, pd.Timestamp]): Sample date
        latitude (float): Sample latitude
        longitude (float): Sample longitude
        save_dir (Path): Directory to save all raw source data
    """
    # Query Copernicus DEM data

    # Save out data for sample
    pass


def download_elevation_data(sample_list: pd.Dataframe, save_dir: Path):
    """Query Copernicus' DEM elevation database for a list of samples, and
    save out the raw results.

    Args:
        sample_list (pd.Dataframe): Dataframe with columns for date,
            longitude, latitude, and uid
        save_dir (Path): Directory in which to save raw elevation data
    """
    logger.info(f"Querying elevation data for {sample_list.shape[0]:,} samples")

    # Iterate over samples (parallelize later)
    for sample in tqdm(sample_list.itertuples()):
        download_sample_elevation(
            sample.Index, sample.date, sample.latitude, sample.longitude, save_dir=save_dir
        )
