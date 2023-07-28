from typing import List, Optional

from pydantic import BaseModel

from cyano.settings import RANDOM_STATE


class LGBParams(BaseModel):
    application: Optional[str] = "regression"
    metric: Optional[str] = "rmse"
    max_depth: Optional[int] = -1
    num_leaves: Optional[int] = 31
    learning_rate: Optional[float] = 0.1
    verbosity: Optional[int] = -1
    bagging_seed: Optional[int] = RANDOM_STATE
    seed: Optional[int] = RANDOM_STATE


class LGBConfig(BaseModel):
    params: Optional[LGBParams] = LGBParams()
    num_boost_round: Optional[int] = 1000


class ExperimentConfig(BaseModel):
    model_save_dir: str
    num_threads: Optional[int] = 5
    cache_dir: Optional[str] = None
    pc_collections: Optional[List] = ["sentinel-2-l2a"]
    pc_days_search_window: Optional[int] = 30
    pc_meters_search_window: Optional[int] = 1000
    use_sentinel_bands: Optional[List] = ["B02", "B03", "B04"]
    image_feature_meter_window: Optional[int] = 500
    satellite_features: Optional[List] = [
        "B02_mean",
        "B02_min",
        "B02_max",
        "B03_mean",
        "B03_min",
        "B03_max",
        "B04_mean",
    ]
    climate_features: Optional[List] = []
    elevation_features: Optional[List] = []
    metadata_features: Optional[List] = []
    lgb_config: Optional[LGBConfig] = LGBConfig()
