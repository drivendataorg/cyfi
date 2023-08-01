import yaml

from loguru import logger
import pandas as pd
from pathlib import Path

from cyano.config import TrainConfig, PredictConfig
from cyano.data.climate_data import download_climate_data
from cyano.data.elevation_data import download_elevation_data
from cyano.data.features import generate_features
from cyano.data.satellite_data import identify_satellite_data, download_satellite_data
from cyano.data.utils import add_unique_identifier
from cyano.models.cyano_model import CyanoModel


def prepare_features(samples, features_config):
    cache_dir = features_config.cache_dir

    ## Identify satellite data
    satellite_meta = identify_satellite_data(samples, features_config)
    save_satellite_to = Path(cache_dir) / "satellite_metadata_train.csv"
    satellite_meta.to_csv(save_satellite_to, index=False)
    logger.info(
        f"{satellite_meta.shape[0]:,} rows of satellite metadata saved to {save_satellite_to}"
    )

    ## Download satellite data
    download_satellite_data(satellite_meta, samples, features_config)

    ## Download non-satellite data
    if features_config.climate_features:
        download_climate_data(samples, features_config)
    if features_config.elevation_features:
        download_elevation_data(samples, features_config)
    logger.success(f"Raw source data saved to {cache_dir}")

    ## Generate features
    features = generate_features(samples, features_config)
    save_features_to = Path(cache_dir) / "features_train.csv"
    features.to_csv(save_features_to, index=True)
    logger.success(
        f"{features.shape[1]:,} features for {features.shape[0]:,} samples saved to {save_features_to}"
    )
    return features


def train_model(labels: pd.DataFrame, config: TrainConfig, debug: bool = False):
    """Train a cyanobacteria prediction model

    Args:
        labels (pd.DataFrame): Dataframe with columns for date, longitude,
            latitude, and severity
        config (TrainConfig): Training configuration
        debug (bool, optional): Whether to run training on only a small
            subset of samples. Defaults to False.
    """
    cache_dir = config.features_config.make_cache_dir()

    ## Load labels
    labels = labels[["date", "latitude", "longitude", "severity"]]
    labels = add_unique_identifier(labels)
    if debug:
        labels = labels.head(10)

    # Save out samples with uids
    labels.to_csv(Path(cache_dir) / "train_samples_uid_mapping.csv", index=True)
    logger.info(f"Loaded {labels.shape[0]:,} samples for training")

    samples = labels[["date", "latitude", "longitude"]]
    labels = labels["severity"]

    ## Query from feature data sources and save
    features = prepare_features(samples, config.features_config)

    ## Instantiate model
    model_config = config.tree_model_config
    model = CyanoModel(model_config)

    ## Train model
    logger.info(f"Training model with LGB params: {model_config}")
    model.train(features, labels)

    ## Save model
    Path(model_config.save_dir).mkdir(exist_ok=True, parents=True)
    logger.info(f"Saving model to {model_config.save_dir}")
    model.save(model_config.save_dir)

    ## Save out config
    with open(f"{model_config.save_dir}/config.yaml", "w") as fp:
        yaml.dump(config.model_dump(), fp)

    return model


def predict_model(samples: pd.DataFrame, config: PredictConfig, debug: bool = False):
    """Load an existing cyanobacteria prediction model from trained_model_dir and generate
    severity level predictions for a set of samples.

    Args:
        samples (pd.DataFrame): Dataframe of samples with columns for date,
            longitude, and latitude
        config (PredictConfig): Prediction configuration
    """
    cache_dir = config.features_config.make_cache_dir()

    ## Load data
    samples = add_unique_identifier(samples)
    if debug:
        samples = samples.head(10)

    # Save out samples with uids
    samples.to_csv(Path(cache_dir) / "predict_samples_uid_mapping.csv", index=True)
    logger.info(f"Loaded {samples.shape[0]:,} samples for prediction")

    ## Query from feature data sources and save
    features = prepare_features(samples, config.features_config)

    ## Load model
    model = CyanoModel.load_model(config.weights)
    logger.info(f"Loaded model from {config.weights}")

    ## Predict and combine with sample info
    preds = model.predict(features)
    samples["predicted_severity"] = preds.loc[samples.index]

    ## Save out predictions
    Path(config.preds_path).parent.mkdir(exist_ok=True, parents=True)
    samples.to_csv(config.preds_path, index=True)
    logger.success(f"Predictions saved to {config.preds_path}")

    return samples
