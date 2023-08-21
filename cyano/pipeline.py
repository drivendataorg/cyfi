from pathlib import Path
import tempfile
from typing import Optional
import yaml
from zipfile import ZipFile

import lightgbm as lgb
from loguru import logger
import pandas as pd

from cyano.config import FeaturesConfig, ModelTrainingConfig
from cyano.data.climate_data import download_climate_data
from cyano.data.elevation_data import download_elevation_data
from cyano.data.features import generate_features
from cyano.data.satellite_data import identify_satellite_data, download_satellite_data
from cyano.data.utils import (
    add_unique_identifier,
    water_distance_filter,
    convert_density_to_severity,
)


class CyanoModelPipeline:
    def __init__(
        self,
        features_config: FeaturesConfig,
        model_training_config: Optional[ModelTrainingConfig] = None,
        cache_dir: Optional[Path] = None,
        model: Optional[lgb.Booster] = None,
        target_col: Optional[str] = "severity",
    ):
        """Instantiate CyanoModelPipeline

        Args:
            features_config (FeaturesConfig): Features configuration
            model_training_config (Optional[ModelTrainingConfig], optional): Model
                training configuration. Defaults to None.
            cache_dir (Optional[Path], optional): Cache directory. Defaults to None.
            model (Optional[lgb.Booster], optional): Trained LGB model. Defaults to None.
            target_col (Optional[str], optional): Target column to predict. Must be
                either "severity" or "density_cells_per_ml". Defaults to "severity".
        """
        self.features_config = features_config
        self.model_training_config = model_training_config
        self.model = model
        self.cache_dir = (
            Path(tempfile.TemporaryDirectory().name) if cache_dir is None else Path(cache_dir)
        )
        self.samples = None
        self.labels = None
        self.target_col = target_col

        # make cache dir
        self.cache_dir.mkdir(exist_ok=True, parents=True)

    def _prep_train_data(self, data, filter_by_water_distance: Optional[int], debug: bool):
        """Load labels and save out samples with UIDs"""
        labels = pd.read_csv(data)
        labels = labels[["date", "latitude", "longitude", self.target_col]]
        labels = add_unique_identifier(labels)
        if debug:
            labels = labels.head(10)

        # Filter by distance to water if specified
        if filter_by_water_distance is not None:
            labels = water_distance_filter(labels, filter_by_water_distance)

        # Save out samples with uids
        labels.to_csv(self.cache_dir / "train_samples_uid_mapping.csv", index=True)
        logger.info(f"Loaded {labels.shape[0]:,} samples for training")

        self.train_samples = labels[["date", "latitude", "longitude"]]
        self.train_labels = labels[self.target_col]

    def _prepare_features(self, samples, train_split=True):
        if train_split:
            split = "train"
        else:
            split = "test"
        ## Identify satellite data
        satellite_meta = identify_satellite_data(samples, self.features_config)
        save_satellite_to = self.cache_dir / f"satellite_metadata_{split}.csv"
        satellite_meta.to_csv(save_satellite_to, index=False)
        logger.info(
            f"{satellite_meta.shape[0]:,} rows of satellite metadata saved to {save_satellite_to}"
        )

        ## Download satellite data
        download_satellite_data(satellite_meta, samples, self.features_config, self.cache_dir)

        ## Download non-satellite data
        if self.features_config.climate_features:
            download_climate_data(samples, self.features_config, self.cache_dir)
        if self.features_config.elevation_features:
            download_elevation_data(samples, self.features_config, self.cache_dir)
        logger.success(f"Raw source data saved to {self.cache_dir}")

        ## Generate features
        features = generate_features(samples, satellite_meta, self.features_config, self.cache_dir)
        save_features_to = self.cache_dir / f"features_{split}.csv"
        features.to_csv(save_features_to, index=True)

        return features

    def _prepare_train_features(self):
        self.train_features = self._prepare_features(self.train_samples)

    def _train_model(self):
        lgb_data = lgb.Dataset(
            self.train_features, label=self.train_labels.loc[self.train_features.index]
        )

        self.model = lgb.train(
            self.model_training_config.params.model_dump(),
            lgb_data,
            num_boost_round=self.model_training_config.num_boost_round,
        )

    def _to_disk(self, save_path: Path):
        save_dir = Path(save_path).parent
        save_dir.mkdir(exist_ok=True, parents=True)

        ## Zip up model config and weights
        logger.info(f"Saving model to {save_path}")
        with ZipFile(save_path, "w") as z:
            z.writestr("config.yaml", yaml.dump(self.features_config.model_dump()))
            z.writestr("lgb_model.txt", self.model.model_to_string())

    def run_training(
        self, train_csv, save_path, filter_by_water_distance: bool = False, debug: bool = False
    ):
        self._prep_train_data(train_csv, filter_by_water_distance, debug)
        self._prepare_train_features()
        self._train_model()
        self._to_disk(save_path)

    @classmethod
    def from_disk(cls, filepath, cache_dir=None):
        archive = ZipFile(filepath, "r")
        features_config = FeaturesConfig(**yaml.safe_load(archive.read("config.yaml")))
        model = lgb.Booster(model_str=archive.read("lgb_model.txt").decode())
        return cls(features_config=features_config, model=model, cache_dir=cache_dir)

    def _prep_predict_data(self, data, debug: bool = False):
        df = pd.read_csv(data)
        df = add_unique_identifier(df)

        samples = df[["date", "latitude", "longitude"]]

        if debug:
            samples = samples.head(10)

        # Save out samples with uids
        samples.to_csv(self.cache_dir / "predict_samples_uid_mapping.csv", index=True)
        logger.info(f"Loaded {samples.shape[0]:,} samples for prediction")

        self.predict_samples = samples

    def _prepare_predict_features(self):
        self.predict_features = self._prepare_features(self.predict_samples, train_split=False)

    def _predict_model(self):
        preds = pd.Series(
            data=self.model.predict(self.predict_features),
            index=self.predict_features.index,
            name="severity",
        )

        # Group by sample id if multiple predictions per id
        if not preds.index.is_unique:
            logger.info(
                f"Grouping {preds.shape[0]:,} predictions by {preds.index.nunique():,} unique sample IDs"
            )
            preds = preds.groupby(preds.index).mean()
        self.preds = preds.round()

        self.output_df = self.predict_samples.join(self.preds)

        # If predicting exact density, calculate severity bin
        if self.target_col == "density_cells_per_ml":
            self.output_df = convert_density_to_severity(self.output_df)

        missing_mask = self.output_df.severity.isna()
        logger.warning(
            f"{missing_mask.sum():,} samples do not have predictions ({missing_mask.mean():.0%})"
        )

    def _write_predictions(self, preds_path):
        Path(preds_path).parent.mkdir(exist_ok=True, parents=True)
        self.output_df.to_csv(preds_path, index=True)
        logger.success(f"Predictions saved to {preds_path}")

    def run_prediction(self, predict_csv, preds_path, debug=False):
        self._prep_predict_data(predict_csv, debug)
        self._prepare_predict_features()
        self._predict_model()
        self._write_predictions(preds_path)
