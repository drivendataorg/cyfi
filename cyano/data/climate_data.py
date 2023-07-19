from loguru import logger
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def download_sample_climate(date, latitude, longitude, save_dir: Path):
    """Query one sample's climate data based on its date, latitude, and longitude, and download the result.

    Args:
        date (_type_): _description_
        latitude (_type_): _description_
        longitude (_type_): _description_
    """
    pass


def download_climate_data(sample_list: pd.Dataframe, save_dir: Path):
    """Query NOAA's HRRR database for a list of samples, and save out the raw results.

    Args:
        sample_list (pd.Dataframe): Dataframe with columns for date,
            longitude, latitude, and uid
        features_dir (Path): Directory in which to save raw climate data
    """
    logger.info(f"Querying climate data for {sample_list.shape[0]:,} samples")

    # Iterate over samples (parallelize later)
    for sample in tqdm(sample_list.itertuples()):
        download_sample_climate(sample.date, sample.latitude, sample.longitude, save_dir=save_dir)
