from pathlib import Path
import tempfile
from typing import List, Optional
import yaml
from repro_zipfile import ReproducibleZipFile as ZipFile

import lightgbm as lgb
from loguru import logger
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from cyano.config import FeaturesConfig, CyanoModelConfig
from cyano.data.features import generate_all_features
from cyano.data.satellite_data import identify_satellite_data, download_satellite_data
from cyano.data.utils import (
    add_unique_identifier,
    convert_density_to_severity,
    convert_density_to_log_density,
    convert_log_density_to_density,
)


class CyanoModelPipeline:
    def __init__(
        self,
        features_config: FeaturesConfig = FeaturesConfig(),
        cyano_model_config: CyanoModelConfig = CyanoModelConfig(),
        cache_dir: Optional[Path] = None,
        models: Optional[List[lgb.Booster]] = None,
    ):
        """Instantiate CyanoModelPipeline

        Args:
            features_config (FeaturesConfig): Features configuration
            cyano_model_config (Optional[CyanoModelConfig], optional): Model
                training configuration. Defaults to None.
            cache_dir (Optional[Path], optional): Cache directory. Defaults to None.
            model (Optional[lgb.Booster], optional): Trained LGB model. Defaults to None.
        """
        self.features_config = features_config
        self.cyano_model_config = cyano_model_config
        self.models = models

        # Determine cache dir based on feature config hash
        cache_dir = Path(tempfile.gettempdir()) if cache_dir is None else Path(cache_dir)
        self.cache_dir = cache_dir / self.features_config.get_cached_path()

        # make cache dir
        self.cache_dir.mkdir(exist_ok=True, parents=True)

        self.samples = None
        self.labels = None
        self.target_col = cyano_model_config.target_col

    def _prep_train_data(self, data, debug: bool):
        """Load labels and save out samples with UIDs"""
        labels = pd.read_csv(data)
        # Check that we have required columns
        for col in ["latitude", "longitude", "date"]:
            if col not in labels:
                raise ValueError(f"Labels dataframe is missing required column {col}")
        if ("log_density" not in labels) and ("density_cells_per_ml" not in labels):
            raise ValueError(
                "Labels dataframe must include a column for either `density_cells_per_ml` or `log_density`"
            )

        labels = add_unique_identifier(labels)
        if debug:
            labels = labels.head(10)

        # Add log density if needed
        if (self.target_col == "log_density") and ("log_density" not in labels):
            labels["log_density"] = convert_density_to_log_density(labels.density_cells_per_ml)
        elif (self.target_col == "density_cells_per_ml") and (
            "density_cells_per_ml" not in labels
        ):
            labels["density_cells_per_ml"] = convert_log_density_to_density(labels.log_density)

        # Save out samples with uids
        labels.to_csv(self.cache_dir / "train_samples_uid_mapping.csv", index=True)
        logger.info(f"Loaded {labels.shape[0]:,} sample(s) for training")

        expected_cols = ["date", "latitude", "longitude"]
        if "region" in labels.columns:
            expected_cols += ["region"]

        self.train_samples = labels[expected_cols]
        self.train_labels = labels[self.target_col]

    def _prepare_features(self, samples, train_split: bool = True):
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
        logger.success(f"Raw satellite imagery saved to {self.cache_dir}")

        ## Generate features
        features = generate_all_features(
            samples, satellite_meta, self.features_config, self.cache_dir
        )
        save_features_to = self.cache_dir / f"features_{split}.csv"
        features.to_csv(save_features_to, index=True)

        return features

    def _prepare_train_features(self):
        self.train_features = self._prepare_features(self.train_samples)

    def _train_model_without_folds(self):
        # Set early stopping to None since we have no validation set
        self.cyano_model_config.params.early_stopping_round = None

        lgb_data = lgb.Dataset(
            self.train_features, label=self.train_labels.loc[self.train_features.index]
        )
        self.models = [
            lgb.train(
                self.cyano_model_config.params.model_dump(),
                lgb_data,
                num_boost_round=self.cyano_model_config.num_boost_round,
            )
        ]

    def _train_model_with_folds(self):
        train_features = self.train_features.copy().reset_index(drop=False)
        kf = StratifiedGroupKFold(
            n_splits=self.cyano_model_config.n_folds,
            shuffle=True,
            random_state=40,
        )
        splits = kf.split(
            train_features,
            self.train_samples.loc[train_features.sample_id].region,
            groups=train_features.sample_id,
        )

        trained_models = []
        for train_idx, valid_idx in splits:
            # Train model on fold
            train_split_features = train_features.loc[train_idx].set_index("sample_id")
            valid_split_features = train_features.loc[valid_idx].set_index("sample_id")

            lgb_train_data = lgb.Dataset(
                train_split_features,
                label=self.train_labels.loc[train_split_features.index],
            )
            lgb_valid_data = lgb.Dataset(
                valid_split_features,
                label=self.train_labels.loc[valid_split_features.index],
                reference=lgb_train_data,
            )

            trained_model = lgb.train(
                self.cyano_model_config.params.model_dump(),
                lgb_train_data,
                valid_sets=[lgb_valid_data],
                valid_names=["valid"],
                num_boost_round=self.cyano_model_config.num_boost_round,
            )
            trained_models.append(trained_model)

        self.models = trained_models

    def _validate_training_with_folds(self):
        """Determine whether a model will be trained with folds. Returns True if training with folds has been specified and is supported."""
        if self.cyano_model_config.n_folds == 1:
            return False

        # Training with folds requires region, check if region has been provided
        elif "region" not in self.train_samples:
            logger.warning(
                f"Ignoring n_folds = {self.cyano_model_config.n_folds} and training without folds because `region` is not in the labels dataframe."
            )
            return False

        # Check if there are not enough samples
        elif self.train_features.index.nunique() <= self.cyano_model_config.n_folds:
            logger.warning(
                f"Ignoring n_folds = {self.cyano_model_config.n_folds} and training without folds because there are not enough samples."
            )
            return False

        # Check if any reion has fewer samples than the number of folds
        elif (
            self.train_samples.loc[self.train_features.index].region.value_counts().min()
            < self.cyano_model_config.n_folds
        ):
            logger.warning(
                f"Ignoring n_folds = {self.cyano_model_config.n_folds} and training without folds because at least one region has fewer than n_folds samples."
            )
            return False

        else:
            return True

    def _train_model(self):
        train_with_folds = self._validate_training_with_folds()

        if train_with_folds:
            # Train with folds, distributing regions evenly between folds
            logger.info(f"Training LGB model with {self.cyano_model_config.n_folds} folds")
            self._train_model_with_folds()
        else:
            logger.info("Training single LGB model")
            self._train_model_without_folds()

    def _to_disk(self, save_path: Path):
        save_dir = Path(save_path).parent
        save_dir.mkdir(exist_ok=True, parents=True)

        ## Zip up model config and weights
        logger.info(f"Saving model to {save_path}")
        with ZipFile(save_path, "w") as z:
            z.writestr("features_config.yaml", yaml.dump(self.features_config.model_dump()))
            z.writestr("cyano_model_config.yaml", yaml.dump(self.cyano_model_config.model_dump()))
            for idx, model in enumerate(self.models):
                z.writestr(f"lgb_model_{idx}.txt", model.model_to_string())

    def run_training(self, train_csv, save_path, debug: bool = False):
        self._prep_train_data(train_csv, debug)
        self._prepare_train_features()
        self._train_model()
        self._to_disk(save_path)

    @classmethod
    def from_disk(cls, filepath, cache_dir=None):
        archive = ZipFile(filepath, "r")
        features_config = FeaturesConfig(**yaml.safe_load(archive.read("features_config.yaml")))
        cyano_model_config = CyanoModelConfig(
            **yaml.safe_load(archive.read("cyano_model_config.yaml"))
        )
        # Determine the number of ensembled models
        model_files = [name for name in archive.namelist() if "lgb_model" in name]
        logger.info(f"Loading {len(model_files)} ensembled models")
        models = []
        for model_file in model_files:
            models.append(lgb.Booster(model_str=archive.read(model_file).decode()))

        return cls(
            features_config=features_config,
            cyano_model_config=cyano_model_config,
            models=models,
            cache_dir=cache_dir,
        )

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
        logger.info(f"Ensembling {len(self.models)} models")
        preds = []
        for model in self.models:
            preds.append(
                pd.Series(
                    data=model.predict(self.predict_features), index=self.predict_features.index
                )
            )
        preds = pd.concat(preds).rename(self.target_col)

        # Group by sample id if multiple predictions per id
        if not preds.index.is_unique:
            logger.info(
                f"Grouping {preds.shape[0]:,} predictions by {preds.index.nunique():,} unique sample IDs"
            )
            preds = preds.groupby(preds.index).mean()

        # do not allow negative values
        preds.loc[preds < 0] = 0

        self.preds = preds
        self.output_df = self.predict_samples.join(self.preds)

        # Convert to exact density if not predicted
        if self.target_col == "log_density":
            self.output_df["density_cells_per_ml"] = convert_log_density_to_density(
                self.output_df.log_density
            )
            self.output_df = self.output_df.drop(columns=["log_density"])

        # Add severity level prediction if not already predicted
        if self.target_col != "severity":
            self.output_df["severity"] = convert_density_to_severity(
                self.output_df.density_cells_per_ml
            )

        # Round density prediction
        self.output_df["density_cells_per_ml"] = self.output_df["density_cells_per_ml"].round()

        missing_mask = self.output_df.severity.isna()
        if missing_mask.any():
            logger.warning(
                f"{missing_mask.sum():,} samples do not have predictions ({missing_mask.mean():.0%})"
            )

    def _write_predictions(self, preds_path):
        Path(preds_path).parent.mkdir(exist_ok=True, parents=True)
        self.output_df.to_csv(preds_path, index=True)
        logger.success(f"Predictions saved to {preds_path}")

    def run_prediction(self, predict_csv, preds_path=None, debug=False):
        self._prep_predict_data(predict_csv, debug)
        self._prepare_predict_features()
        self._predict_model()
        if preds_path is not None:
            self._write_predictions(preds_path)
