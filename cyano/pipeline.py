from pathlib import Path
import tempfile
from typing import Optional
import yaml
from zipfile import ZipFile

import lightgbm as lgb
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from cyano.config import FeaturesConfig, ModelTrainingConfig
from cyano.data.climate_data import download_climate_data
from cyano.data.elevation_data import download_elevation_data
from cyano.data.features import generate_features
from cyano.data.satellite_data import identify_satellite_data, download_satellite_data
from cyano.data.utils import add_unique_identifier
from cyano.settings import RANDOM_STATE


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
        self.cache_dir = (
            Path(tempfile.TemporaryDirectory().name) if cache_dir is None else Path(cache_dir)
        )
        self.samples = None
        self.labels = None

        # make cache dir
        self.cache_dir.mkdir(exist_ok=True, parents=True)

    def _prep_train_data(self, data, debug=False):
        """Load labels and save out samples with UIDs"""
        labels = pd.read_csv(data)
        # labels = labels[["date", "latitude", "longitude", "severity"]]
        labels = add_unique_identifier(labels)
        if debug:
            labels = labels.head(10)

        # Save out samples with uids
        labels.to_csv(self.cache_dir / "train_samples_uid_mapping.csv", index=True)
        logger.info(f"Loaded {labels.shape[0]:,} samples for training")

        self.train_samples = labels  # [["date", "latitude", "longitude"]]
        self.train_labels = labels["severity"]

    def _prepare_features(self, samples):
        ## Identify satellite data
        satellite_meta = identify_satellite_data(samples, self.features_config)
        save_satellite_to = self.cache_dir / "satellite_metadata_train.csv"
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
        save_features_to = self.cache_dir / "features_train.csv"
        features.to_csv(save_features_to, index=True)
        logger.success(
            f"{features.shape[1]:,} features for {features.index.nunique():,} samples (of {samples.shape[0]:,}) saved to {save_features_to}"
        )

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

    def _train_single_model(self):
        lgb_data = lgb.Dataset(
            self.train_features, label=self.train_labels.loc[self.train_features.index]
        )

        return lgb.train(
            self.model_training_config.params.model_dump(),
            lgb_data,
            num_boost_round=self.model_training_config.num_boost_round,
        )

    def _train_model_with_folds(self):
        n_folds = 5
        # Train without folds if we cannot split folds by region
        if "region" not in self.train_samples.columns:
            self.models = [self._train_single_model()]
            return

        min_in_region = (
            self.train_samples.loc[self.train_features.index].region.value_counts().min()
        )
        if min_in_region < n_folds:
            self.models = [self._train_single_model()]
            return

        # Train with folds by region
        logger.info(f"Training with {n_folds} grouped by region")
        train_features = self.train_features.copy()
        train_features = train_features.reset_index(drop=False)
        train_features["fold"] = np.nan
        kf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
        splits = kf.split(
            train_features,
            self.train_samples.loc[train_features.sample_id].region,
            groups=train_features.sample_id,
        )

        trained_models = []
        for fold_id, (train_idx, valid_idx) in enumerate(splits):
            # train_features.loc[test_idx, "fold"] = fold_id

            # Train model on fold
            train_split_features = train_features.loc[train_idx].set_index("sample_id")
            valid_split_features = train_features.loc[valid_idx].set_index("sample_id")

            lgb_train_data = lgb.Dataset(
                train_split_features, label=self.train_labels.loc[train_split_features.index]
            )
            lgb_valid_data = lgb.Dataset(
                valid_split_features,
                label=self.train_labels.loc[valid_split_features.index],
                reference=lgb_train_data,
            )

            evals_result = {}
            trained_model = lgb.train(
                self.model_training_config.params.model_dump(),
                lgb_train_data,
                valid_sets=[lgb_train_data, lgb_valid_data],
                valid_names=["train", "valid"],
                eval_results=evals_result,
                um_boost_round=self.model_training_config.num_boost_round,
                early_stopping_rounds=self.model_training_config.early_stopping_rounds,
            )
            trained_models.append(trained_model)

            # valid_preds = trained_model.predict(valid_split_features)
            # TODO: maybe ad dlogging of performance on valid set here
            # or saving of eval results

        self.models = trained_models

    def _to_disk(self, save_path: Path):
        save_dir = Path(save_path).parent
        save_dir.mkdir(exist_ok=True, parents=True)

        ## Zip up model config and weights
        logger.info(f"Saving model to {save_path}")
        with ZipFile(save_path, "w") as z:
            z.writestr("config.yaml", yaml.dump(self.features_config.model_dump()))
            z.writestr("lgb_model.txt", self.model.model_to_string())

    def run_training(self, train_csv, save_path, debug=False):
        self._prep_train_data(train_csv, debug)
        self._prepare_train_features()
        self._train_model()
        self._to_disk(save_path)

    @classmethod
    def from_disk(cls, filepath, cache_dir=None):
        archive = ZipFile(filepath, "r")
        features_config = FeaturesConfig(**yaml.safe_load(archive.read("config.yaml")))
        model = lgb.Booster(model_str=archive.read("lgb_model.txt").decode())
        return cls(features_config=features_config, model=model, cache_dir=cache_dir)

    def _prep_predict_data(self, data, debug=False):
        df = add_unique_identifier(pd.read_csv(data))

        samples = df[["date", "latitude", "longitude"]]

        if debug:
            samples = samples.head(10)

        # Save out samples with uids
        samples.to_csv(self.cache_dir / "predict_samples_uid_mapping.csv", index=True)
        logger.info(f"Loaded {samples.shape[0]:,} samples for prediction")

        self.predict_samples = samples

    def _prepare_predict_features(self):
        self.predict_features = self._prepare_features(self.predict_samples)

    def _predict_model(self):
        preds = []
        for model in self.models:
            preds.append(
                pd.Series(
                    data=model.predict(self.predict_features),
                    index=self.predict_features.index,
                    name="severity",
                )
            )

        preds = pd.concat(preds)
        # preds = pd.Series(
        #     data=self.model.predict(self.predict_features),
        #     index=self.predict_features.index,
        #     name="severity",
        # )

        # Group by sample id if multiple predictions per id
        if not preds.index.is_unique:
            logger.info(
                f"Grouping {preds.shape[0]:,} predictions from {len(self.models)} by {preds.index.nunique():,} unique sample IDs"
            )
            preds = preds.groupby(preds.index).mean()
        self.preds = preds.round().astype(int)

        self.output_df = self.predict_samples.join(self.preds)
        # For any missing samples, predict the average predicted severity
        logger.info(
            f"Predicting the mean for {self.output_df.severity.isna().sum():,} samples with no imagery"
        )
        self.output_df["severity"] = self.output_df.severity.fillna(preds.mean().round()).astype(
            int
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
