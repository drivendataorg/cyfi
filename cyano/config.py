from typing import List, Optional

from pydantic import BaseModel, ConfigDict

from cyano.settings import RANDOM_STATE


class LGBParams(BaseModel):
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
    """LightGBM model training parameters. For details, see
    [LightGBM's documentation](https://lightgbm.readthedocs.io/en/latest/Parameters.html).

    Args:
        application (Optional[str], optional): Regression application. Defaults to "regression".
        metric (Optional[str], optional): Metric to use. Defaults to "rmse".
        max_depth (Optional[int], optional): Limit the max depth for tree model. Defaults to -1.
        num_leaves (Optional[int], optional): Max number of tree leaves. Defaults to 31.
        learning_rate (Optional[float], optional): Learning rate. Defaults to 0.1.
        verbosity (Optional[int], optional): Level of LightGBM's verbosity. Defaults to -1.
        bagging_seed (Optional[int], optional): Random seed for bagging. Defaults to RANDOM_STATE.
        seed (Optional[int], optional): Seed used to generate other random seeds. Defaults
            to RANDOM_STATE.

    """


class FeaturesConfig(BaseModel):
    pc_collections: Optional[List] = ["sentinel-2-l2a"]
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
    climate_features: Optional[List] = []
    elevation_features: Optional[List] = []
    metadata_features: Optional[List] = ["rounded_longitude"]
    scl_filter: Optional[bool] = False
    """Features configuration

    Args:
        pc_collections (Optional[List], optional): Collections within the planetary computer to
            search for satellite imagery. Defaults to ["sentinel-2-l2a"].
        pc_days_search_window (Optional[int], optional): Number of days before a given sample was
            collected to include when searching the planetary computer. Defaults to 30.
        pc_meters_search_window (Optional[int], optional): Buffer in meters to add on each side
            of a given sample's location when searching the planetary computer for satellite imagery.
            Defaults to 1000.
        use_sentinel_bands (Optional[List], optional): All Sentinel-2 bands that will be used when
            generating satellite imagery featues. Defaults to ["B02", "B03", "B04"].
        image_feature_meter_window (Optional[int], optional): Buffer in meters on each side of a
            given sample's location to use when creating the bounding box for the subset of a satellite
            image that will be used to generate features. Defaults to 500.
        n_sentinel_items (Optional[int], optional): Maximum number of Sentinel-2 items to use for each
            sample. Defaults to 1.
        satellite_image_features (Optional[List], optional): List of satellite imagery features to
            include. Defaults to [ "B02_mean", "B02_min", "B02_max", "B03_mean", "B03_min",
            "B03_max", "B04_mean", ].
        satellite_meta_features (Optional[List], optional): List of satellite metadata features to
            include. Defaults to [].
        climate_features (Optional[List], optional): List of climate features to include.
            Defaults to [].
        elevation_features (Optional[List], optional): List of elevation features to include.
            Defaults to [].
        metadata_features (Optional[List], optional): List of metadata features to include. Defaults
            to ["rounded_longitude"].
        scl_filter (Optional[bool], optional): Whether to filter satellite imagery during feature
            generation to pixels labeled as water by Sentinel-2's SCL band. Defaults to False.
    """


class ModelTrainingConfig(BaseModel):
    params: Optional[LGBParams] = LGBParams()
    num_boost_round: Optional[int] = 1000
    n_folds: Optional[int] = 1

    # Silence warning for conflict with pydantic protected namespace
    model_config = ConfigDict(protected_namespaces=())
    """Model training configuration

    Args:
        params (Optional[LGBParams], optional): Parameters for LightGBM training. For details
            see [LightGBM's documentation](https://lightgbm.readthedocs.io/en/latest/Parameters.html).
            Defaults to LGBParams().s
        num_boost_round (Optional[int], optional): Number of boosting iterations. Defaults to 1000.
    """
