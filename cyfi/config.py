import hashlib
from typing import List, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from pydantic.types import confloat


def check_field_is_subset(field_value: List, accepted_values: List) -> List:
    """Check that a list-list field value is a subset of the accepted values

    Args:
        field_value (List): Field value
        accepted_values (list): Accepted values for the given field
    """
    unrecognized = np.setdiff1d(field_value, accepted_values)
    if unrecognized.size > 0:
        raise ValueError(
            f"Unrecognized value(s): {list(unrecognized)}. Possible values are: {accepted_values}"
        )

    return field_value


class FeaturesConfig(BaseModel):
    """Features configuration

    Args:
        pc_days_search_window (Optional[int], optional): Number of days before a given sample was
            collected to include when searching the planetary computer for satellite imagery.
            Defaults to 30.
        pc_meters_search_window (Optional[int], optional): Buffer in meters to add on each side
            of a given sample's location when searching the planetary computer for satellite imagery.
            Defaults to 2000.
        use_sentinel_bands (Optional[List], optional): All Sentinel-2 bands that are needed to
            generate satellite imagery featues. For all options see AVAILABLE_SENTINEL_BANDS.
            Defaults to ['AOT', 'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09',
            'B11', 'B12', 'B8A', 'SCL', 'WVP'].
        image_feature_meter_window (Optional[int], optional): Buffer in meters to add on each side of a
            given sample's location when creating the bounding box for the subset of a satellite image
            that will be used to generate features. Defaults to 2000.
        max_cloud_percent (Optional[float], optional): Maximum portion of cloud pixels allowed in bounding box.
            If the portion of cloud pixels is above this, the image will not be used. Defaults to 0.05.
            Ranges from 0-1.
        filter_to_water_area (bool): Whether to filter to water pixels in the bounding box using the scene
            classification band. Defaults to True.
        n_sentinel_items (Optional[int], optional): Maximum number of Sentinel-2 items to download for each
            sample. Defaults to 15. Only the most recent one containing water and passing the max_cloud_percent
            filter (if using) will be used to generate features.
        satellite_image_features (Optional[List], optional): List of satellite imagery features to
            include. For all options see SATELLITE_FEATURE_CALCULATORS. Defaults to ['AOT_mean',
            'AOT_min', 'AOT_max', 'AOT_range', 'B01_mean', 'B01_min', 'B01_max', 'B01_range',
            'B02_mean', 'B02_min', 'B02_max', 'B02_range', 'B03_mean', 'B03_min', 'B03_max',
            'B03_range', 'B04_mean', 'B04_min', 'B04_max', 'B04_range', 'B05_mean', 'B05_min',
            'B05_max', 'B05_range', 'B06_mean', 'B06_min', 'B06_max', 'B06_range', 'B07_mean',
            'B07_min', 'B07_max', 'B07_range', 'B08_mean', 'B08_min', 'B08_max', 'B08_range',
            'B09_mean', 'B09_min', 'B09_max', 'B09_range', 'B11_mean', 'B11_min', 'B11_max',
            'B11_range', 'B12_mean', 'B12_min', 'B12_max', 'B12_range', 'B8A_mean', 'B8A_min',
            'B8A_max', 'B8A_range', 'SCL_mean', 'SCL_min', 'SCL_max', 'SCL_range', 'WVP_mean',
            'WVP_min', 'WVP_max', 'WVP_range', 'NDVI_B04', 'NDVI_B05', 'NDVI_B06', 'NDVI_B07'].
        satellite_meta_features (Optional[List], optional): List of satellite metadata features to
            include. For all options see AVAILABLE_SATELLITE_META_FEATURSE. Defaults to ['month',
            'days_before_sample'].
        sample_meta_features (Optional[List], optional): List of metadata features to include. For all
            options see AVAILABLE_SAMPLE_META_FEATURES. Defaults to ['land_cover'].
    """

    pc_days_search_window: Optional[int] = 30
    pc_meters_search_window: Optional[int] = 2000
    use_sentinel_bands: Optional[List] = [
        "AOT",
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B09",
        "B11",
        "B12",
        "B8A",
        "SCL",
        "WVP",
    ]
    image_feature_meter_window: Optional[int] = 2000
    max_cloud_percent: Optional[confloat(ge=0, le=1)] = 0.05
    filter_to_water_area: bool = True
    n_sentinel_items: Optional[int] = 15
    satellite_meta_features: Optional[List] = ["month", "days_before_sample"]
    sample_meta_features: Optional[List] = ["land_cover"]
    satellite_image_features: Optional[List] = [
        "B01_mean",
        "B02_mean",
        "B03_mean",
        "B04_mean",
        "B05_mean",
        "B06_mean",
        "B07_mean",
        "B08_mean",
        "B09_mean",
        "B11_mean",
        "B12_mean",
        "B8A_mean",
        "WVP_mean",
        "AOT_mean",
        "percent_water",
        "green95th",
        "green5th",
        "green_red_ratio",
        "green_blue_ratio",
        "red_blue_ratio",
        "green95th_blue_ratio",
        "green5th_blue_ratio",
        "NDVI_B04",
        "NDVI_B05",
        "NDVI_B06",
        "NDVI_B07",
        "AOT_range",
    ]

    # Do not allow extra fields
    model_config = ConfigDict(extra="forbid")

    @field_validator("use_sentinel_bands")
    def validate_sentinel_bands(cls, field):
        return check_field_is_subset(field, AVAILABLE_SENTINEL_BANDS)

    @model_validator(mode="after")
    def include_SCL_for_water_or_cloud_filtering(self):
        if (
            self.max_cloud_percent is not None or self.filter_to_water_area
        ) and "SCL" not in self.use_sentinel_bands:
            # add SCL which is used for water and cloud filtering
            self.use_sentinel_bands = self.use_sentinel_bands + ["SCL"]
        return self

    @field_validator("satellite_image_features")
    def validate_satellite_image_features(cls, field):
        return check_field_is_subset(field, list(SATELLITE_FEATURE_CALCULATORS.keys()))

    @field_validator("satellite_meta_features")
    def validate_satellite_meta_features(cls, field):
        return check_field_is_subset(field, AVAILABLE_SATELLITE_META_FEATURES)

    @field_validator("sample_meta_features")
    def validate_sample_meta_features(cls, field):
        return check_field_is_subset(field, AVAILABLE_SAMPLE_META_FEATURES)

    def get_cached_path(self) -> str:
        """Get the hash used for the features cache directory name"""
        config_dict = self.model_dump()

        # Only include keys that change the saved image arrays
        # Apply consistent sorting
        config_dict_to_hash = {
            "image_feature_meter_window": config_dict["image_feature_meter_window"],
            "use_sentinel_bands": sorted(config_dict["use_sentinel_bands"]),
        }

        # Get hash
        hash_str = hashlib.md5(str(config_dict_to_hash).encode()).hexdigest()
        return hash_str


class LGBParams(BaseModel):
    """LightGBM model training parameters. For details, see
    [LightGBM's documentation](https://lightgbm.readthedocs.io/en/latest/Parameters.html).

    Args:
        application (Optional[str], optional): Regression application. Defaults to "regression".
        metric (Optional[str], optional): Metric to use. Defaults to "rmse".
        max_depth (Optional[int], optional): Limit the max depth for tree model. Defaults to -1.
        num_leaves (Optional[int], optional): Max number of tree leaves. Defaults to 31.
        learning_rate (Optional[float], optional): Learning rate. Defaults to 0.1.
        verbosity (Optional[int], optional): Level of LightGBM's verbosity. Defaults to -1.
        feature_fraction (Optional[float], optional): If smaller than 1.0, LightGBM will
            randomly select this percentage subset of features on each iteration before
            training. Defaults to 0.6
        early_stopping_round (Optional[int], optional): If provided, stop training if one
            metric of one validation data doesn't improve in the last `early_stopping_round`
            rounds. Defaults to 100.
        seed (Optional[int], optional): Seed used to generate other random seeds. Defaults
            to 40.
    """

    application: Optional[str] = "regression"
    metric: Optional[str] = "rmse"
    max_depth: Optional[int] = -1
    num_leaves: Optional[int] = 31
    learning_rate: Optional[float] = 0.1
    verbosity: Optional[int] = -1
    feature_fraction: Optional[float] = 0.6
    early_stopping_round: Optional[int] = 100
    seed: Optional[int] = 40

    # Do not allow extra fields
    model_config = ConfigDict(extra="forbid")


class CyFiModelConfig(BaseModel):
    """Model configuration

    Args:
        params (Optional[LGBParams], optional): Parameters for LightGBM training. For details
            see [LightGBM's documentation](https://lightgbm.readthedocs.io/en/latest/Parameters.html).
            Defaults to LGBParams().
        num_boost_round (Optional[int], optional): Number of boosting iterations. Defaults to 100000.
        n_folds (Optional[int], optional): Number of different model folds to train. If greater than
            5, the models will be ensembled for a final prediction. Defaults to 1.
        target_col (Optional[str], optional):  Target column to predict. For possible
            values, see AVAILABLE_TARGET_COLS. Defaults to "log_density".
    """

    params: Optional[LGBParams] = LGBParams()
    num_boost_round: Optional[int] = 100_000
    n_folds: Optional[int] = 5
    target_col: Optional[str] = "log_density"

    # Do not allow extra fields
    # Silence warning for conflict with pydantic protected namespace
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    @field_validator("target_col")
    def validate_target_col(cls, field):
        return check_field_is_subset(field, AVAILABLE_TARGET_COLS)


AVAILABLE_TARGET_COLS = ["severity", "log_density", "density_cells_per_ml"]

AVAILABLE_SENTINEL_BANDS = [
    "AOT",
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B10",
    "B11",
    "B12",
    "SCL",
    "WVP",
]

AVAILABLE_SATELLITE_META_FEATURES = [
    "month",
    "days_before_sample",
    "eo:cloud_cover",
    "s2:nodata_pixel_percentage",
    "min_long",
    "max_long",
    "min_lat",
    "max_lat",
]

AVAILABLE_SAMPLE_META_FEATURES = ["land_cover", "rounded_latitude", "rounded_longitude"]

SATELLITE_FEATURE_CALCULATORS = {
    "NDVI_B04": lambda x: (x["B08"].mean() - x["B04"].mean())
    / (x["B08"].mean() + x["B04"].mean()),
    "NDVI_B05": lambda x: (x["B08"].mean() - x["B05"].mean())
    / (x["B08"].mean() + x["B05"].mean()),
    "NDVI_B06": lambda x: (x["B08"].mean() - x["B06"].mean())
    / (x["B08"].mean() + x["B06"].mean()),
    "NDVI_B07": lambda x: (x["B08"].mean() - x["B07"].mean())
    / (x["B08"].mean() + x["B07"].mean()),
    "green_red_ratio": lambda x: x["B03"].mean() / (x["B04"].mean()),
    "green_blue_ratio": lambda x: x["B03"].mean() / (x["B02"].mean()),
    "red_blue_ratio": lambda x: x["B04"].mean() / (x["B02"].mean()),
    "green95th": lambda x: np.percentile(x["B03"], 95),
    "green5th": lambda x: np.percentile(x["B03"], 5),
    "green95th_blue_ratio": lambda x: np.percentile(x["B03"], 95) / (x["B02"].mean()),
    "green5th_blue_ratio": lambda x: np.percentile(x["B03"], 5) / (x["B02"].mean()),
    "percent_water": lambda x: (x["SCL"] == 6).mean(),
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
