import functools
import os
from typing import List

from cloudpathlib import AnyPath
from herbie import FastHerbie
from loguru import logger
import numpy as np
import pandas as pd
from pandas_path import path  # noqa
from tqdm.contrib.concurrent import process_map

from cyano.config import FeaturesConfig

TRY_GRID_BUFFERS = np.arange(start=0.01, stop=0.1, step=0.01)


def path_to_climate_data(sample_id, var_name, level, cache_dir):
    """TODO: describe organization of climate data files


    Args:
        sample_id (_type_): _description_
        var_name (_type_): _description_
        level (_type_): _description_
        cache_dir (_type_): _description_

    Returns:
        _type_: _description_
    """
    return AnyPath(f"{cache_dir}/{var_name.lower()}_{level.replace(' ', '_')}/{sample_id}.csv")


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


def query_grid_mapping(
    samples: pd.DataFrame, cache_dir, example_dates: List = None
) -> pd.DataFrame:
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
    logger.info(
        f"Iterating over {locations.shape[0]:,} locations to get HRRR grid indices with {NUM_PROCESSES} workers"
    )
    results = []
    for chunk in np.array_split(locations, 10):
        chunk_results = process_map(
            functools.partial(get_location_grid, xarrays_df=xarrays_df),
            chunk.latitude,
            chunk.longitude,
            max_workers=NUM_PROCESSES,
            chunksize=1,
            total=len(chunk),
        )

        results += chunk_results
        interim_results = pd.concat([res for res in results if res is not None]).reset_index(
            drop=True
        )
        interim_results.to_csv(cache_dir / "interim_hrrr_location_grid_mapping.csv", index=False)

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
            missing_samples, cache_dir, example_dates=herbie_example_dates
        )
        sample_grid_map = pd.concat([sample_grid_map, new_sample_grid_map])

    # Otherwise generate grid for all samples
    else:
        logger.info(f"Generating HRRR grid for {samples.shape[0]:,} samples")
        sample_grid_map = query_grid_mapping(samples, cache_dir)

    logger.info(
        f"Saving updated grid mapping with {sample_grid_map.index.nunique():,} samples to {grid_save_path}"
    )
    sample_grid_map["date"] = pd.to_datetime(sample_grid_map.date)
    sample_grid_map.to_csv(grid_save_path, index=True)

    return sample_grid_map[sample_grid_map.index.isin(samples.index)]


def get_timestamps_per_date(dates):
    query_timestamps = []
    sample_dates = []
    for date in dates:
        timestamps = [
            d for d in pd.date_range(start=date, periods=12, freq="-1H", inclusive="both")
        ] + [d for d in pd.date_range(start=date, periods=12, freq="1H", inclusive="both")]
        timestamps = list(set(timestamps))
        query_timestamps += timestamps
        sample_dates += [date for i in timestamps]

    return (
        pd.DataFrame({"query_timestamp": query_timestamps, "sample_date": sample_dates})
        .drop_duplicates()
        .sort_values(by=["sample_date", "query_timestamp"])
        .reset_index(drop=True)
    )


def download_climate_for_date(
    date,
    timestamps_per_date: pd.DataFrame,
    sample_grid_map: pd.DataFrame,
    cache_dir,
    var_name: str,
    level: str,
):
    # First, check if samples for given date are already downloaded
    date_sample_paths = (
        sample_grid_map[sample_grid_map.date == date].reset_index().sample_id.drop_duplicates()
    )
    date_sample_paths = date_sample_paths.apply(
        functools.partial(
            path_to_climate_data, var_name=var_name, level=level, cache_dir=cache_dir
        )
    )
    if date_sample_paths.path.exists().all():
        return

    # Otherwise query for climate data
    try:
        query_timestamps = timestamps_per_date[
            timestamps_per_date.sample_date == date
        ].query_timestamp.tolist()
        FH = FastHerbie(query_timestamps, model="hrrr", fxx=[0], verbose=False)
        xarray_data = FH.xarray(f"{var_name}:{level}", remove_grib=False).drop_vars(
            ["step", "time", "longitude", "latitude"]
        )
        xarray_df = xarray_data.to_dataframe().reset_index()

        # Filter to grid locations that we need for the given date
        xarray_df = xarray_df.merge(
            sample_grid_map[sample_grid_map.date == date].reset_index(), on=["x", "y"], how="inner"
        )
    except Exception as e:
        logger.warning(f"{type(e)}: {e} for {date}")
        return

    # Save out data by sample id
    for sample_id in xarray_df.sample_id.unique():
        sample_data = xarray_df[xarray_df.sample_id == sample_id].set_index("sample_id")
        sample_data_path = path_to_climate_data(sample_id, var_name, level, cache_dir)
        sample_data.to_csv(sample_data_path, index=True)


def download_climate_data(samples: pd.DataFrame, config: FeaturesConfig, cache_dir):
    """Query NOAA's HRRR database for a list of samples, and save out
    the raw results.

    Args:
        samples (pd.Dataframe): Dataframe with columns for date,
            longitude, latitude, and sample_id
        config (FeaturesConfig): Configuration, including
            directory to save raw source data
    """
    # Determine the number of processes to use when parallelizing
    NUM_PROCESSES = int(os.getenv("CY_NUM_PROCESSES", 4))

    logger.info(f"Querying climate data for {samples.shape[0]:,} samples")
    samples = process_samples_for_hrrr(samples)

    ## Get mapping of sample locations to grid indices in HRRR
    sample_grid_map = load_hrrr_grid(samples, cache_dir)
    logger.info(
        f"Loaded HRRR grid mapping for {sample_grid_map.index.nunique():,} samples ({sample_grid_map.shape[0]:,} rows total)"
    )

    ## Download climate data for each date
    dates = samples.date.unique()
    timestamps_per_date = get_timestamps_per_date(dates)
    for climate_source in config.use_climate_sources:
        var_name, level = climate_source
        logger.info(f"Downloading {var_name} at {level} for {len(dates):,} dates")
        process_map(
            functools.partial(
                download_climate_for_date,
                timestamps_per_date=timestamps_per_date,
                sample_grid_map=sample_grid_map,
                cache_dir=cache_dir,
                var_name=var_name,
                level=level,
            ),
            dates,
            max_workers=NUM_PROCESSES,
            chunksize=1,
            total=len(dates),
        )
