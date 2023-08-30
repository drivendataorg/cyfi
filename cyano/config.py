from typing import List, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator

from cyano.settings import RANDOM_STATE, AVAILABLE_SENTINEL_BANDS, SATELLITE_FEATURE_CALCULATORS


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
            training. Defaults to 1.0.
        early_stopping_round (Optional[int], optional): If provided, stop training if one
            metric of one validation data doesn't improve in the last `early_stopping_round`
            rounds. Defaults to None.
        bagging_seed (Optional[int], optional): Random seed for bagging. Defaults to RANDOM_STATE.
        seed (Optional[int], optional): Seed used to generate other random seeds. Defaults
            to RANDOM_STATE.
    """

    application: Optional[str] = "regression"
    metric: Optional[str] = "rmse"
    max_depth: Optional[int] = -1
    num_leaves: Optional[int] = 31
    learning_rate: Optional[float] = 0.1
    verbosity: Optional[int] = -1
    feature_fraction: Optional[float] = 1.0
    early_stopping_round: Optional[int] = None
    bagging_seed: Optional[int] = RANDOM_STATE
    seed: Optional[int] = RANDOM_STATE


def check_field_is_subset(field_value: List, accepted_values: List) -> List:
    """Check that a list-list field value is a subset of the accepted values

    Args:
        field_value (List): Field value
        accepted_values (list): Accepted values for the given field
    """
    unrecognized = np.setdiff1d(field_value, accepted_values)
    if unrecognized:
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
            Defaults to 1000.
        use_sentinel_bands (Optional[List], optional): All Sentinel-2 bands that are needed to
            generate satellite imagery featues. Defaults to ["B02", "B03", "B04"].
        image_feature_meter_window (Optional[int], optional): Buffer in meters to add on each side of a
            given sample's location when creating the bounding box for the subset of a satellite image
            that will be used to generate features. Defaults to 500.
        n_sentinel_items (Optional[int], optional): Maximum number of Sentinel-2 items to use for each
            sample. Defaults to 1.
        satellite_image_features (Optional[List], optional): List of satellite imagery features to
            include. Defaults to [ "B02_mean", "B02_min", "B02_max", "B03_mean", "B03_min",
            "B03_max", "B04_mean"].
        satellite_meta_features (Optional[List], optional): List of satellite metadata features to
            include. Defaults to [].
        metadata_features (Optional[List], optional): List of metadata features to include. Defaults
            to [].
    """

    pc_days_search_window: Optional[int] = 30
    pc_meters_search_window: Optional[int] = 1000
    use_sentinel_bands: Optional[List] = ["B02", "B03", "B04"]
    image_feature_meter_window: Optional[int] = 500
    n_sentinel_items: Optional[int] = 1
    satellite_image_features: Optional[List] = [
        "B02_mean",
        "B02_min",
        "B02_max",
        "B03_mean",
        "B03_min",
        "B03_max",
        "B04_mean",
    ]
    satellite_meta_features: Optional[List] = []
    metadata_features: Optional[List] = []

    @field_validator("use_sentinel_bands")
    def validate_sentinel_bands(cls, path_field):
        return check_field_is_subset(path_field, AVAILABLE_SENTINEL_BANDS)

    @field_validator("satellite_image_features")
    def validate_satellite_image_features(cls, path_field):
        return check_field_is_subset(path_field, list(SATELLITE_FEATURE_CALCULATORS.keys()))

    @field_validator("satellite_meta_features")
    def validate_satellite_meta_features(cls, path_field):
        return check_field_is_subset(
            path_field,
            [
                "month",
                "days_before_sample",
                "eo:cloud_cover",
                "s2:nodata_pixel_percentage",
                "min_long",
                "max_long",
                "min_lat",
                "max_lat",
            ],
        )

    @field_validator("metadata_features")
    def validate_metadata_features(cls, path_field):
        return check_field_is_subset(
            path_field, ["land_cover", "rounded_latitude", "rounded_longitude"]
        )


class ModelTrainingConfig(BaseModel):
    """Model training configuration

    Args:
        params (Optional[LGBParams], optional): Parameters for LightGBM training. For details
            see [LightGBM's documentation](https://lightgbm.readthedocs.io/en/latest/Parameters.html).
            Defaults to LGBParams().
        num_boost_round (Optional[int], optional): Number of boosting iterations. Defaults to 1000.
        n_folds (Optional[int], optional): Number of different model folds to train. If greater than
            1, the models will be ensembled for a final prediction. Defaults to 1.
    """

    params: Optional[LGBParams] = LGBParams()
    num_boost_round: Optional[int] = 1000
    n_folds: Optional[int] = 1

    # Silence warning for conflict with pydantic protected namespace
    model_config = ConfigDict(protected_namespaces=())
