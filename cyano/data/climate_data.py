import functools
import os

from cloudpathlib import AnyPath
from herbie import FastHerbie
from loguru import logger
import numpy as np
import pandas as pd
from pandas_path import path  # noqa
from tqdm.contrib.concurrent import process_map

from cyano.config import FeaturesConfig

TRY_GRID_BUFFERS = np.arange(start=0.01, stop=0.1, step=0.01)

# Save consistent example dates to initialize FastHerbie
# Include a range to ensure we get grid mapping results for all samples
HERBIE_EXAMPLE_DATES = [
    d for d in pd.date_range(start="2014-09-01", end="2022-01-01", freq="6m", inclusive="both")
]


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


def query_grid_mapping(samples: pd.DataFrame, cache_dir) -> pd.DataFrame:
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
    FH = FastHerbie(HERBIE_EXAMPLE_DATES, model="hrrr", fxx=[0], verbose=False)
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
    for chunk in np.array_split(locations, max(1, len(locations) // 1000)):
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
            sample_grid_map["date"] = pd.to_datetime(sample_grid_map.date)
            return sample_grid_map.loc[samples.index]

        new_sample_grid_map = query_grid_mapping(missing_samples, cache_dir)
        sample_grid_map = pd.concat([sample_grid_map, new_sample_grid_map]).drop_duplicates()

    # Otherwise generate grid for all samples
    else:
        sample_grid_map = query_grid_mapping(samples, cache_dir).drop_duplicates()

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
    # Query for climate data
    try:
        query_timestamps = timestamps_per_date[
            timestamps_per_date.sample_date.dt.date == date.date()
        ].query_timestamp.tolist()
        FH = FastHerbie(query_timestamps, model="hrrr", fxx=[0], verbose=False)
        xarray_data = FH.xarray(f"{var_name}:{level}", remove_grib=False).drop_vars(
            ["step", "time", "longitude", "latitude"]
        )
        xarray_df = xarray_data.to_dataframe().reset_index()

        # Filter to grid locations that we need for the given date
        xarray_df = xarray_df.merge(
            sample_grid_map[sample_grid_map.date.dt.date == date.date()].reset_index(),
            on=["x", "y"],
            how="inner",
        )
    except Exception as e:
        logger.warning(f"{type(e)}: {e} for date {date}")
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

    ## Check which samples are missing expected climate files
    samples = process_samples_for_hrrr(samples)
    for climate_var in config.climate_variables:
        newcol = f"exists_{climate_var}_{config.climate_level}"
        samples[newcol] = samples.index.map(
            functools.partial(
                path_to_climate_data,
                var_name=climate_var,
                level=config.climate_level,
                cache_dir=cache_dir,
            )
        )
        samples[newcol] = samples[newcol].path.exists()
    missing_any_climate_files_mask = ~samples.filter(regex="exists_").all(axis=1)

    if missing_any_climate_files_mask.sum() == 0:
        logger.success(
            f"All required climate files have already been downloaded for sources {config.climate_variables}"
        )
        return

    samples = samples[missing_any_climate_files_mask].copy()
    logger.info(
        f"Getting climate data for {samples.shape[0]:,} samples for sources {config.climate_variables}"
    )

    ## Get mapping of sample locations to grid indices in HRRR
    sample_grid_map = load_hrrr_grid(samples, cache_dir)

    ## Iterate over required climate data sources
    for climate_var in config.climate_variables:
        # Create source / level directory
        path_to_climate_data(
            "sample_id", climate_var, config.climate_level, cache_dir
        ).parent.mkdir(exist_ok=True, parents=True)

        # Check which samples are missing files for this climate source
        missing_files_mask = samples[f"exists_{climate_var}_{config.climate_level}"] != True
        missing_dates = samples[missing_files_mask].date.unique()
        timestamps_per_date = get_timestamps_per_date(missing_dates)
        logger.info(
            f"Downloading {climate_var} at {config.climate_level} for {len(missing_dates):,} date(s)"
        )
        # Download climate data for each date
        _ = process_map(
            functools.partial(
                download_climate_for_date,
                timestamps_per_date=timestamps_per_date,
                sample_grid_map=sample_grid_map,
                cache_dir=cache_dir,
                var_name=climate_var,
                level=config.climate_level,
            ),
            missing_dates,
            max_workers=NUM_PROCESSES,
            chunksize=1,
            total=len(missing_dates),
        )
