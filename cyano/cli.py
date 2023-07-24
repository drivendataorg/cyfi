import json
from typing import Dict, Optional

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
    labels_path: Path,
    config_path: Optional[Dict] = None,
):
    """Train a cyanobacteria prediction model

    Args:
        labels_path (Path): Path to a csv with columns for date,
            longitude, latitude, and severity
        config (Dict): Experiment config
    """
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config path does not exist: {config_path}")

    with open(config_path, "r") as fp:
        config = json.load(fp)

    ## Load labels
    labels = pd.read_csv(labels_path)
    labels = add_unique_identifier(labels)
    logger.info(f"Loaded {labels.shape[0]:,} samples for training")

    ## Query from feature data sources and save
    samples = labels[["date", "latitude", "longitude"]]

    satellite_meta = identify_satellite_data(samples, config)
    save_satellite_to = Path(config["features_dir"]) / "satellite_metadata_train.csv"
    satellite_meta.to_csv(save_satellite_to, index=False)
    logger.info(
        f"{satellite_meta.shape[0]:,} rows of satellite metadata saved to {save_satellite_to}"
    )
    download_satellite_data(satellite_meta, config, samples)
    download_climate_data(samples, config)
    download_elevation_data(samples, config)
    logger.success(f"Raw source data saved to {config['features_dir']}")

    ## Generate features
    features = generate_features(samples, config)
    save_features_to = Path(config["model_dir"]) / "all_features_train.csv"
    features.to_csv(save_features_to, index=True)
    logger.info(
        f"Generated {features.shape[1]:,} features for {features.shape[0]:,} samples. Saved to {save_features_to}"
    )

    ## Instantiate model
    model = CyanoModel(config)

    ## Train model and save
    logger.info(f"Training model with LGB params: {model.config['lgb_params']}")
    model.train(features, labels)

    logger.info(f"Saving model to {config['model_dir']}")
    model.save(config["model_dir"])


@app.command()
def predict(sample_list_path: Path, preds_save_path: Path, model_dir: Path):
    """Load an existing cyanobacteria prediction model and generate
    severity level predictions for a set of samples.

    Args:
        sample_list_path (Path): Path to a csv with columns for date,
            longitude, and latitude
        preds_save_path (Path): Path to save the generated predictions
        model_dir (Path): Directory containing model weights and
            experiment configuration
    """
    ## Load model and experiment config
    model = CyanoModel.load_model(model_dir)
    logger.info(f"Loaded model from {model_dir} with lgb params {model.config['lgb_params']}")
    config = model.config

    ## Load data
    samples = pd.read_csv(sample_list_path)
    samples = add_unique_identifier(samples)
    logger.info(f"Loaded {samples.shape[0]:,} samples for prediction")

    ## Query from feature data sources and save
    satellite_meta = identify_satellite_data(samples, config)
    save_satellite_to = model_dir / "satellite_metadata_predict.csv"
    satellite_meta.to_csv(save_satellite_to, index=False)
    logger.info(f"Satellite metadata saved to {save_satellite_to}")
    download_satellite_data(satellite_meta, config)

    download_climate_data(samples, config)
    download_elevation_data(samples, config)
    logger.success(f"Raw source data saved to {config['features_dir']}")

    ## Generate features
    features = generate_features(samples, config, satellite_meta)
    save_features_to = model_dir / "all_features_predict.csv"
    features.to_csv(save_features_to, index=True)
    logger.info(
        f"Generated {features.shape[1]:,} features for {features.shape[0]:,} samples. Saved to {save_features_to}"
    )

    ## Predict
    preds = model.predict(features)
    preds = preds.join(samples)  # Add sample info
    preds.to_csv(preds_save_path, index=True)
    logger.success(f"Predictions saved to {preds_save_path}")


if __name__ == "__main__":
    app()
