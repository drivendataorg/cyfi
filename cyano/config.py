from pathlib import Path
import tempfile
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

class FeaturesConfig(BaseModel):
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

    def make_cache_dir(self):
        """Create cache directory for features.
        Creates a temp directory if no cache dir is specified.

        Returns the cache_dir location.
        """
        if self.cache_dir is None:
            self.cache_dir = tempfile.TemporaryDirectory().name
        Path(self.cache_dir).mkdir(exist_ok=True, parents=True)
        return self.cache_dir

class ModelConfig(BaseModel):
    params: Optional[LGBParams] = LGBParams()
    num_boost_round: Optional[int] = 1000
    # TODO: make these paths
    save_dir: str = "cyano_model"

class TrainConfig(BaseModel):
    num_threads: Optional[int] = 5
    features_config: FeaturesConfig = FeaturesConfig()
    tree_model_config: ModelConfig = ModelConfig()

class PredictConfig(BaseModel):
    save_path: str
    features_config: FeaturesConfig
    tree_model_config: ModelConfig

# a model has a weights file and config (in form of ModelConfig)
# TODO: make model save dir an absolute path so we can use it to load