from loguru import logger
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# retuen to
# - selectin speciifc items
# - query through an specifying args


def download_sample_imagery(
    date, latitude, longitude, sources, time_window, geographic_window
) -> pd.Dataframe:
    """Query one sample based on its date, latitude, and longitude, and download the result.

    Args:
        date (_type_): _description_
        latitude (_type_): _description_
        longitude (_type_): _description_
        sources (_type_): _description_
        time_window (_type_): _description_
        geographic_window (_type_): _description_

    Returns:
        pd.Dataframe: Dataframe listing all possible PySTAC items
            to be included for the given sample
    """

    # Query planetary computer
    return pd.DataFrame()


def select_items(items: pd.DataFrame):
    pass


def download_satellite_data(sample_list: pd.Dataframe, features_dir: Path):
    """Query the planetary computer for Sentinel-2 and Landsat satellite
    data, and save out the raw results.

    Args:
        sample_list (pd.Dataframe): Dataframe with columns for date,
            longitude, latitude, and uid
        features_dir (Path): Directory in which to save raw satellite data
    """
    logger.info(f"Querying satellite data for {sample_list.shape[0]:,} samples")

    # Iterate over samples (parallelize later)
    for sample in tqdm(sample_list.itertuples()):
        download_sample_imagery(sample.date, sample.latitude, sample.longitude)

    pass
