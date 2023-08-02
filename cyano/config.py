from pathlib import Path
import tempfile
from typing import List, Optional
import yaml
from zipfile import ZipFile

from loguru import logger
from pydantic import BaseModel, PrivateAttr

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
    save_dir: str = "cyano_model"
    _model = PrivateAttr()


class TrainConfig(BaseModel):
    features_config: FeaturesConfig = FeaturesConfig()
    tree_model_config: ModelConfig = ModelConfig()

    def sanitize_features_config(self):
        """Sanitize for use for weights"""
        data = self.model_dump()
        data["features_config"].pop("cache_dir")
        return data["features_config"]

    def save_model(self):
        """Save out zipfile with model weights and features config along with artifact config.
        """
        model_config = self.tree_model_config

        Path(model_config.save_dir).mkdir(exist_ok=True, parents=True)

        ## Save out model weights
        model_config._model.lgb_model.save_model(Path(model_config.save_dir) / "lgb_model.txt")

        ## Save features config associated with weights
        with open(f"{model_config.save_dir}/config.yaml", "w") as fp:
            yaml.dump(self.sanitize_features_config(), fp)

        ## Zip up model config and weights and keep only zip file
        logger.info(f"Saving model zip to {model_config.save_dir}")
        with ZipFile(f"{model_config.save_dir}/model.zip", "w") as z:
            for fp in [
                f"{model_config.save_dir}/lgb_model.txt",
                f"{model_config.save_dir}/config.yaml",
            ]:
                z.write(fp, Path(fp).name)

                Path(fp).unlink()

        ## Save artifact config
        with open(f"{model_config.save_dir}/config_artifact.yaml", "w") as fp:
            yaml.dump(self.model_dump(), fp)


class PredictConfig(BaseModel):
    features_cache_dir: Optional[str] = None
    weights_zipfile: str
    preds_path: str = "preds.csv"
