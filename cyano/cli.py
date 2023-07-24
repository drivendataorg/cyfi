import json
from typing import Dict

from loguru import logger
import pandas as pd
from pathlib import Path
import typer

from cyano.data.climate_data import download_climate_data
from cyano.data.elevation_data import download_elevation_data
from cyano.data.features import generate_features
from cyano.data.satellite_data import identify_satellite_data, download_satellite_data
from cyano.data.utils import add_unique_identifier
from cyano.models.cyano_model import CyanoModel

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def train(
    labels_path: Path = typer.Argument(
        exists=True, help="Path to a csv with columns for date, longitude, latitude, and severity"
    ),
    config_path: Path = typer.Argument(exists=True, help="Path to an experiment configuration"),
    debug: bool = typer.Option(
        False, help="Whether to run training on only a small subset of samples"
    ),
):
    """Train a cyanobacteria prediction model based on the labels in labels_path
    and the config file saved at config_path. The trained model and full experiment
    configuration will be saved to the "model_dir" specified in the config
    """
    with open(config_path, "r") as fp:
        config = json.load(fp)

    labels = pd.read_csv(labels_path)

    train_model(labels, config, debug=debug)


def train_model(labels: pd.DataFrame, config: Dict, debug: bool = False):
    """Train a cyanobacteria prediction model

    Args:
        labels (pd.DataFrame): Dataframe with columns for date, longitude,
            latitude, and severity
        config (Dict): Experiment configuration
        debug (bool, optional): Whether to run training on only a small
            subset of samples. Defaults to False.
    """
    Path(config["model_dir"]).mkdir(exist_ok=True, parents=True)

    ## Load labels
    labels = add_unique_identifier(labels)
    if debug:
        labels = labels.head(10)
    logger.info(f"Loaded {labels.shape[0]:,} samples for training")

    ## Query from feature data sources and save
    samples = labels[["date", "latitude", "longitude"]]
    labels = labels["severity"]

    satellite_meta = identify_satellite_data(samples, config)
    save_satellite_to = Path(config["model_dir"]) / "satellite_metadata_train.csv"
    satellite_meta.to_csv(save_satellite_to, index=False)
    logger.info(
        f"{satellite_meta.shape[0]:,} rows of satellite metadata saved to {save_satellite_to}"
    )
    download_satellite_data(satellite_meta, samples, config)
    # download_climate_data(samples, config)
    # download_elevation_data(samples, config)
    logger.success(f"Raw source data saved to {config['features_dir']}")

    ## Generate features
    features = generate_features(samples, config)
    save_features_to = Path(config["model_dir"]) / "features_train.csv"
    features.to_csv(save_features_to, index=True)
    logger.success(
        f"Generated {features.shape[1]:,} features for {features.shape[0]:,} samples. Saved to {save_features_to}"
    )

    ## Instantiate model
    model = CyanoModel(config)

    ## Train model and save
    logger.info(f"Training model with LGB params: {model.config['lgb_params']}")
    model.train(features, labels)

    logger.info(f"Saving model to {config['model_dir']}")
    model.save(config["model_dir"])

    return model


@app.command()
def predict(
    samples_path: Path = typer.Argument(
        exists=True, help="Path to a csv of samples with columns for date, longitude, and latitude"
    ),
    model_dir: Path = typer.Argument(
        exists=True, help="Path to directory with saved model weights and config"
    ),
    preds_save_path: Path = typer.Argument(help="Destination to save predictions csv"),
    debug: bool = typer.Option(
        False, help="Whether to generate predictions for only a small subset of samples"
    ),
):
    """Load an existing cyanobacteria prediction model from model_dir and generate
    severity level predictions for a set of samples."""
    samples = pd.read_csv(samples_path)

    predict_model(samples, preds_save_path=preds_save_path, model_dir=model_dir, debug=debug)


def predict_model(
    samples: pd.DataFrame, preds_save_path: Path, model_dir: Path, debug: bool = False
):
    """Load an existing cyanobacteria prediction model from model_dir and generate
    severity level predictions for a set of samples.

    Args:
        samples (pd.DataFrame): Dataframe of samples with columns for date,
            longitude, and latitude
        preds_save_path (Path): Path to save the generated predictions
        model_dir (Path): Directory containing model weights and
            experiment configuration
    """
    model_dir = Path(model_dir)

    ## Load model and experiment config
    model = CyanoModel.load_model(model_dir)
    logger.info(f"Loaded model from {model_dir} with lgb params {model.config['lgb_params']}")
    config = model.config

    ## Load data
    samples = add_unique_identifier(samples)
    if debug:
        samples = samples.head(10)
    logger.info(f"Loaded {samples.shape[0]:,} samples for prediction")

    ## Query from feature data sources and save
    satellite_meta = identify_satellite_data(samples, config)
    save_satellite_to = model_dir / "satellite_metadata_predict.csv"
    satellite_meta.to_csv(save_satellite_to, index=False)
    logger.info(
        f"{satellite_meta.shape[0]:,} rows of satellite metadata saved to {save_satellite_to}"
    )
    download_satellite_data(satellite_meta, samples, config)
    # download_climate_data(samples, config)
    # download_elevation_data(samples, config)
    logger.success(f"Raw source data saved to {config['features_dir']}")

    ## Generate features
    features = generate_features(samples, config)
    save_features_to = model_dir / "features_predict.csv"
    features.to_csv(save_features_to, index=True)
    logger.info(
        f"Generated {features.shape[1]:,} features for {features.shape[0]:,} samples. Saved to {save_features_to}"
    )

    ## Predict and combine with sample info
    preds = model.predict(features)
    samples["predicted_severity"] = preds.loc[samples.index]
    samples.to_csv(preds_save_path, index=True)
    logger.success(f"Predictions saved to {preds_save_path}")

    return samples


if __name__ == "__main__":
    app()
