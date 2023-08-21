## Code to generate features from raw downloaded source data
import functools
import os
from typing import List, Union

from cloudpathlib import AnyPath
import cv2
from loguru import logger
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.contrib.concurrent import process_map

from cyano.config import FeaturesConfig
from cyano.data.climate_data import path_to_climate_data

# Create a dictionary mapping feature names to feature generator
# functions, which take a dictionary of band arrays as input
SATELLITE_FEATURE_CALCULATORS = {
    "NDVI_B04": lambda x: (x["B08"].mean() - x["B04"].mean())
    / (x["B08"].mean() + x["B04"].mean() + 1),
    "NDVI_B05": lambda x: (x["B08"].mean() - x["B05"].mean())
    / (x["B08"].mean() + x["B05"].mean() + 1),
    "NDVI_B06": lambda x: (x["B08"].mean() - x["B06"].mean())
    / (x["B08"].mean() + x["B06"].mean() + 1),
    "NDVI_B07": lambda x: (x["B08"].mean() - x["B07"].mean())
    / (x["B08"].mean() + x["B07"].mean() + 1),
    "green_red_ratio": lambda x: x["B03"].mean() / (x["B04"].mean() + 1),
    "green_blue_ratio": lambda x: x["B03"].mean() / (x["B02"].mean() + 1),
    "red_blue_ratio": lambda x: x["B04"].mean() / (x["B02"].mean() + 1),
    "green95th_blue_ratio": lambda x: np.percentile(x["B03"], 95) / (x["B02"].mean() + 1),
    "green5th_blue_ratio": lambda x: np.percentile(x["B03"], 5) / (x["B02"].mean() + 1),
    "prop_water": lambda x: (x["SCL"] == 6).mean(),
    "AOT_mean": lambda x: x["AOT"].mean(),
    "AOT_min": lambda x: x["AOT"].min(),
    "AOT_max": lambda x: x["AOT"].max(),
    "AOT_range": lambda x: x["AOT"].max() - x["AOT"].min(),
    "B01_mean": lambda x: x["B01"].mean(),
    "B01_min": lambda x: x["B01"].min(),
    "B01_max": lambda x: x["B01"].max(),
    "B01_range": lambda x: x["B01"].max() - x["B01"].min(),
    "B02_mean": lambda x: x["B02"].mean(),
    "B02_min": lambda x: x["B02"].min(),
    "B02_max": lambda x: x["B02"].max(),
    "B02_range": lambda x: x["B02"].max() - x["B02"].min(),
    "B03_mean": lambda x: x["B03"].mean(),
    "B03_min": lambda x: x["B03"].min(),
    "B03_max": lambda x: x["B03"].max(),
    "B03_range": lambda x: x["B03"].max() - x["B03"].min(),
    "B03_95th": lambda x: np.percentile(x["B03"], 95),
    "B03_5th": lambda x: np.percentile(x["B03"], 5),
    "B04_mean": lambda x: x["B04"].mean(),
    "B04_min": lambda x: x["B04"].min(),
    "B04_max": lambda x: x["B04"].max(),
    "B04_range": lambda x: x["B04"].max() - x["B04"].min(),
    "B05_mean": lambda x: x["B05"].mean(),
    "B05_min": lambda x: x["B05"].min(),
    "B05_max": lambda x: x["B05"].max(),
    "B05_range": lambda x: x["B05"].max() - x["B05"].min(),
    "B06_mean": lambda x: x["B06"].mean(),
    "B06_min": lambda x: x["B06"].min(),
    "B06_max": lambda x: x["B06"].max(),
    "B06_range": lambda x: x["B06"].max() - x["B06"].min(),
    "B07_mean": lambda x: x["B07"].mean(),
    "B07_min": lambda x: x["B07"].min(),
    "B07_max": lambda x: x["B07"].max(),
    "B07_range": lambda x: x["B07"].max() - x["B07"].min(),
    "B08_mean": lambda x: x["B08"].mean(),
    "B08_min": lambda x: x["B08"].min(),
    "B08_max": lambda x: x["B08"].max(),
    "B08_range": lambda x: x["B08"].max() - x["B08"].min(),
    "B09_mean": lambda x: x["B09"].mean(),
    "B09_min": lambda x: x["B09"].min(),
    "B09_max": lambda x: x["B09"].max(),
    "B09_range": lambda x: x["B09"].max() - x["B09"].min(),
    "B11_mean": lambda x: x["B11"].mean(),
    "B11_min": lambda x: x["B11"].min(),
    "B11_max": lambda x: x["B11"].max(),
    "B11_range": lambda x: x["B11"].max() - x["B11"].min(),
    "B12_mean": lambda x: x["B12"].mean(),
    "B12_min": lambda x: x["B12"].min(),
    "B12_max": lambda x: x["B12"].max(),
    "B12_range": lambda x: x["B12"].max() - x["B12"].min(),
    "B8A_mean": lambda x: x["B8A"].mean(),
    "B8A_min": lambda x: x["B8A"].min(),
    "B8A_max": lambda x: x["B8A"].max(),
    "B8A_range": lambda x: x["B8A"].max() - x["B8A"].min(),
    "SCL_mean": lambda x: x["SCL"].mean(),
    "SCL_min": lambda x: x["SCL"].min(),
    "SCL_max": lambda x: x["SCL"].max(),
    "SCL_range": lambda x: x["SCL"].max() - x["SCL"].min(),
    "WVP_mean": lambda x: x["WVP"].mean(),
    "WVP_min": lambda x: x["WVP"].min(),
    "WVP_max": lambda x: x["WVP"].max(),
    "WVP_range": lambda x: x["WVP"].max() - x["WVP"].min(),
}

# Mapping of climate variable name to column name its saved out in
CLIMATE_VAR_TO_COL_MAPPING = {"TMP": "t2m", "SPFH": "sh2"}


def generate_features_for_sample_item(
    sample_id: str, item_id: str, config: FeaturesConfig, cache_dir: AnyPath
):
    """Generate the satellite features for specific combination of
    sample and pystac item ID
    """
    sample_item_dir = (
        Path(cache_dir) / f"sentinel_{config.image_feature_meter_window}/{sample_id}/{item_id}"
    )

    # Skip combos we were not able to download
    if not sample_item_dir.exists():
        return None
    # Filter by water pixels from SCL band
    if config.scl_filter:
        scl_band_path = sample_item_dir / "SCL.npy"
        if not scl_band_path.exists():
            return None
        scl_array = np.load(scl_band_path)
        # Skip if there is not enough water
        if (scl_array == 6).mean() < 0.01:
            return None

    # Load band arrays into a dictionary with band names for keys
    band_arrays = {}
    # If we want to mask image data with water boundaries in some way, add here
    for band in config.use_sentinel_bands:
        if not (sample_item_dir / f"{band}.npy").exists():
            raise FileNotFoundError(
                f"Band {band} is missing from pystac item directory {sample_item_dir}"
            )
        # Rescale SCL band based on image size
        band_arr = np.load(sample_item_dir / f"{band}.npy")
        if config.scl_filter and (band != "SCL"):
            scaled_scl = cv2.resize(scl_array[0], (band_arr.shape[2], band_arr.shape[1]))
            # Filter array to water area
            band_arrays[band] = band_arr[0][scaled_scl == 6]
        else:
            band_arrays[band] = band_arr

    # Iterate over features to generate
    sample_item_features = {"sample_id": sample_id, "item_id": item_id}
    for feature in config.satellite_image_features:
        sample_item_features[feature] = SATELLITE_FEATURE_CALCULATORS[feature](band_arrays)

    return sample_item_features


def generate_satellite_features(
    satellite_meta: pd.DataFrame,
    config: FeaturesConfig,
    cache_dir: Union[str, Path],
) -> pd.DataFrame:
    """Generate features from satellite data

    Args:
        satellite_meta (pd.DataFrame): Dataframe of satellite metadata
            for all pystac items that have been selected for use in
            feature generation
        config (FeaturesConfig): Configuration, including
            directory where raw source data is saved
        cache_dir (Union[str, Path]): Cache directory where raw imagery
            is saved

    Returns:
        pd.DataFrame: Dataframe where the index is sample ID and there is one column
            for each satellite feature. There will only be rows for samples with
            satellite imagery. Each row is a unique combination of sample ID and
            item ID
    """
    # Determine the number of processes to use when parallelizing
    NUM_PROCESSES = int(os.getenv("CY_NUM_PROCESSES", 4))

    # Calculate satellite metadata features
    if "month" in config.satellite_meta_features:
        satellite_meta["month"] = pd.to_datetime(satellite_meta.datetime).dt.month

    logger.info(
        f"Generating satellite features for {len(satellite_meta):,} sample/item combos, {satellite_meta.sample_id.nunique():,} samples with {NUM_PROCESSES} processes"
    )

    # Iterate over selected sample / item combinations
    satellite_features = process_map(
        functools.partial(generate_features_for_sample_item, config=config, cache_dir=cache_dir),
        satellite_meta.sample_id,
        satellite_meta.item_id,
        chunksize=1,
        total=len(satellite_meta),
        max_workers=NUM_PROCESSES,
    )

    satellite_features = pd.DataFrame([features for features in satellite_features if features])

    # Add in satellite meta features
    satellite_features = satellite_features.merge(
        satellite_meta[["item_id", "sample_id"] + config.satellite_meta_features],
        how="left",
        on=["item_id", "sample_id"],
        validate="1:1",
    )

    # Check that each row is a unique item / sample combo
    if satellite_features[["sample_id", "item_id"]].duplicated().any():
        raise ValueError(
            "There are repeat sample / item combinations in the satellite features dataframe"
        )

    return satellite_features.set_index("sample_id").drop(columns=["item_id"]).astype(float)


def generate_climate_features(
    sample_ids: Union[List[str], pd.Index], config: FeaturesConfig, cache_dir
) -> pd.DataFrame:
    """Generate features from climate data

    Args:
        sample_ids (Union[List[str], pd.Index]): List of unique indices for each sample
        config (FeaturesConfig): Configuration, including
            directory where raw source data is saved
        cache_dir

    Returns:
        pd.DataFrame: Dataframe where the index is sample_id. There is
            one column for each climate feature and one row for each sample.
            Only samples with climate data are included.
    """
    climate_features = {}
    logger.info(f"Generating climate features for {len(sample_ids):,} samples.")

    for sample_id in tqdm(sample_ids):
        climate_features[sample_id] = {}
        for climate_var in config.climate_variables:
            sample_data_path = path_to_climate_data(
                sample_id, climate_var, config.climate_level, cache_dir
            )
            if not sample_data_path.exists():
                continue

            sample_data = pd.read_csv(sample_data_path)
            var_col_name = CLIMATE_VAR_TO_COL_MAPPING[climate_var]

            if f"{climate_var}_min" in config.climate_features:
                climate_features[sample_id][f"{climate_var}_min"] = sample_data[var_col_name].min()
            if f"{climate_var}_mean" in config.climate_features:
                climate_features[sample_id][f"{climate_var}_mean"] = sample_data[
                    var_col_name
                ].mean()
            if f"{climate_var}_max" in config.climate_features:
                climate_features[sample_id][f"{climate_var}_max"] = sample_data[var_col_name].max()

    climate_features = pd.DataFrame(climate_features).T

    # Drop any rows with no climate features
    missing_mask = climate_features.isna().all(axis=1)

    return climate_features[~missing_mask]


def generate_elevation_features(
    sample_ids: Union[List[str], pd.Index], config: FeaturesConfig
) -> pd.DataFrame:
    """Generate features from elevation data

    Args:
        sample_ids (Union[List[str], pd.Index]): List of unique indices for each sample
        config (FeaturesConfig): Configuration, including
            directory where raw source data is saved

    Returns:
        pd.DataFrame: Dataframe where the index is sample_id. There is
            one columns for each elevation feature and one row
            for each sample
    """
    # Load files
    # - filter to those containing '_elevation' in the name or other pattern
    # - identify data for each sample based on uid

    # Generate features for each sample
    pass


def generate_metadata_features(samples: pd.DataFrame, config: FeaturesConfig) -> pd.DataFrame:
    """Generate features from sample metadata

    Args:
        samples (pd.DataFrame): Dataframe where the index is sample_id and there are
            columns for date, longitude, and latitude
        config (FeaturesConfig): Feature configuration

    Returns:
        pd.DataFrame: Dataframe where the index is sample_id. There is one column
            for each metadata feature and one row for each sample
    """
    # Pull in any external information needed (eg land use by state)

    # Generate features for each sample
    metadata_features = samples.copy()
    if "rounded_longitude" in config.metadata_features:
        metadata_features["rounded_longitude"] = (metadata_features.longitude / 10).round(0)
    if "rounded_latitude" in config.metadata_features:
        metadata_features["rounded_latitude"] = (metadata_features.latitude / 10).round(0)

    return metadata_features[config.metadata_features]


def generate_features(
    samples: pd.DataFrame,
    satellite_meta: pd.DataFrame,
    config: FeaturesConfig,
    cache_dir: Union[str, Path],
) -> pd.DataFrame:
    """Generate a dataframe of features for the given set of samples.
    Requires that the raw satellite, climate, and elevation data for
    the given samples are already saved in cache_dir

    Args:
        samples (pd.DataFrame): Dataframe where the index is sample_id and there are
            columns for date, longitude, and latitude
        satellite_meta (pd.DataFrame): Dataframe of satellite metadata
            for all pystac items that have been selected for use in
            feature generation
        config (FeaturesConfig): Feature configuration
        cache_dir (Union[str, Path]): Cache directory where raw imagery
            is saved

    Returns:
        pd.DataFrame: Dataframe where the index is sample_id and there is one
            column for each feature. Each row is a unique combination of
            sample and pystac item. Samples with *any* features present from a
            source other than the metadata are included. Missing values are left
            as np.nan
    """
    # Generate satellite features
    # May be >1 row per sample, only includes samples with imagery
    satellite_features = generate_satellite_features(satellite_meta, config, cache_dir)
    sample_ct = satellite_features.index.nunique()
    logger.info(
        f"Generated {satellite_features.shape[1]} satellite features for {sample_ct:,} samples ({(sample_ct / samples.shape[0]):.0%})"
    )
    features = satellite_features.copy()

    # Generate non-satellite features. Each has only one row per sample
    sample_ids = samples.index.unique()
    if config.climate_features:
        climate_features = generate_climate_features(sample_ids, config, cache_dir)
        logger.info(
            f"Generated climate features for {climate_features.shape[0]:,} samples ({(climate_features.shape[0] / samples.shape[0]):.0%})"
        )
        features = features.merge(
            climate_features, left_index=True, right_index=True, how="outer", validate="m:1"
        )

    if config.elevation_features:
        elevation_features = generate_elevation_features(sample_ids, config, cache_dir)
        logger.info(
            f"Generated elevation features for {elevation_features.shape[0]:,} samples ({(elevation_features.shape[0] / samples.shape[0]):.0%})"
        )
        features = features.merge(
            elevation_features, left_index=True, right_index=True, how="outer", validate="m:1"
        )

    if config.metadata_features:
        metadata_features = generate_metadata_features(samples, config)
        logger.info(
            f"Generated metadata features for {metadata_features.shape[0]:,} samples ({(metadata_features.shape[0] / samples.shape[0]):.0%})"
        )
        # Don't include samples for which we only have metadata
        features = features.merge(
            metadata_features, left_index=True, right_index=True, how="left", validate="m:1"
        )

    pct_with_features = features.index.nunique() / samples.shape[0]
    logger.success(
        f"Generated {features.shape[1]:,} features for {features.index.nunique():,} samples ({pct_with_features:.0%}). {features.shape[0]:,} rows total"
    )

    all_feature_cols = (
        config.satellite_image_features
        + config.satellite_meta_features
        + config.climate_features
        + config.elevation_features
        + config.metadata_features
    )

    return features[all_feature_cols]
