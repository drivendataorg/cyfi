import tempfile
from typing import Optional
import yaml
from zipfile import ZipFile

import lightgbm as lgb
from loguru import logger
import pandas as pd
from pathlib import Path

from cyano.config import FeaturesConfig, ModelTrainingConfig
from cyano.data.climate_data import download_climate_data
from cyano.data.elevation_data import download_elevation_data
from cyano.data.features import generate_features
from cyano.data.satellite_data import identify_satellite_data, download_satellite_data
from cyano.data.utils import add_unique_identifier


## TODO: add lru_cache

class CyanoModelPipeline:
    def __init__(
        self,
        features_config: FeaturesConfig,
        model_training_config: Optional[ModelTrainingConfig] = None,
        cache_dir: Optional[Path] = None,
        model: Optional[lgb.Booster] = None,
    ):
        self.features_config = features_config
        self.model_training_config = model_training_config
        self.model = model
        self.cache_dir = tempfile.TemporaryDirectory().name if cache_dir is None else cache_dir
        self.samples = None
        self.labels = None

        # make cache dir
        Path(self.cache_dir).mkdir(exist_ok=True, parents=True)

    def prep_train_data(self, data, debug=False):
        """Load labels and save out samples with UIDs"""
        ## Load labels
        labels = pd.read_csv(data)
        labels = labels[["date", "latitude", "longitude", "severity"]]
        labels = add_unique_identifier(labels)
        if debug:
            labels = labels.head(10)

        # Save out samples with uids
        labels.to_csv(Path(self.cache_dir) / "train_samples_uid_mapping.csv", index=True)
        logger.info(f"Loaded {labels.shape[0]:,} samples for training")

        self.samples = labels[["date", "latitude", "longitude"]]
        self.labels = labels["severity"]

        return self.samples, self.labels

    def prepare_features(self):
        if self.samples is None:
            raise ValueError("No samples found")

        ## Identify satellite data
        satellite_meta = identify_satellite_data(
            self.samples, self.features_config, self.cache_dir
        )
        save_satellite_to = Path(self.cache_dir) / "satellite_metadata_train.csv"
        satellite_meta.to_csv(save_satellite_to, index=False)
        logger.info(
            f"{satellite_meta.shape[0]:,} rows of satellite metadata saved to {save_satellite_to}"
        )

        ## Download satellite data
        download_satellite_data(satellite_meta, self.samples, self.features_config, self.cache_dir)

        ## Download non-satellite data
        if self.features_config.climate_features:
            download_climate_data(self.samples, self.features_config, self.cache_dir)
        if self.features_config.elevation_features:
            download_elevation_data(self.samples, self.features_config, self.cache_dir)
        logger.success(f"Raw source data saved to {self.cache_dir}")

        ## Generate features
        features = generate_features(self.samples, self.features_config, self.cache_dir)
        save_features_to = Path(self.cache_dir) / "features_train.csv"
        features.to_csv(save_features_to, index=True)
        logger.success(
            f"{features.shape[1]:,} features for {features.shape[0]:,} samples saved to {save_features_to}"
        )

        self.features = features
        return features

    def train_model(self):
        lgb_data = lgb.Dataset(self.features, label=self.labels.loc[self.features.index])

        ## Train model
        self.model = lgb.train(
            self.model_training_config.params.model_dump(),
            lgb_data,
            num_boost_round=self.model_training_config.num_boost_round,
        )

        return self.model

    def to_disk(self, save_path):
        ## Zip up model config and weights
        save_dir = Path(save_path).parent
        save_dir.mkdir(exist_ok=True, parents=True)

        logger.info(f"Saving model to {save_path}")
        with ZipFile(f"{save_path}", "w") as z:
            z.writestr("config.yaml", yaml.dump(self.features_config.model_dump()))
            z.writestr("lgb_model.txt", self.model.model_to_string())

    def run_training(self, train_csv, save_path="model.zip", debug=False):
        self.prep_train_data(train_csv, debug)
        self.prepare_features()
        self.train_model()
        self.to_disk(save_path)

    @classmethod
    def from_disk(cls, filepath, cache_dir=None):
        archive = ZipFile(filepath, "r")
        features_config = FeaturesConfig(**yaml.safe_load(archive.read("config.yaml")))
        weights_file = archive.extract("lgb_model.txt", "/tmp/weights.txt")
        model = lgb.Booster(model_file=weights_file)
        return cls(features_config=features_config, model=model, cache_dir=cache_dir)

    def prep_predict_data(self, data, debug=False):
        samples = pd.read_csv(data)[["date", "latitude", "longitude"]]

        samples = add_unique_identifier(samples)
        if debug:
            samples = samples.head(10)

        # Save out samples with uids
        samples.to_csv(Path(self.cache_dir) / "predict_samples_uid_mapping.csv", index=True)
        logger.info(f"Loaded {samples.shape[0]:,} samples for prediction")

        self.samples = samples

    def predict_model(self):
        self.preds = pd.Series(
            data=self.model.predict(self.features),
            index=self.features.index,
            name="predicted_severity",
        )
        self.output_df = self.samples.join(self.preds)
        return self.preds

    def write_predictions(self, preds_path):
        ## Save out predictions
        Path(preds_path).parent.mkdir(exist_ok=True, parents=True)
        self.output_df.to_csv(preds_path, index=True)
        logger.success(f"Predictions saved to {preds_path}")

    def run_prediction(self, predict_csv, preds_path="preds.csv"):
        self.prep_predict_data(predict_csv)
        self.prepare_features()
        self.predict_model()
        self.write_predictions(preds_path)
