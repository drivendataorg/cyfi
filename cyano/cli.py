from typing import Dict

from loguru import logger
from pathlib import Path
import typer

from cyano.data.climate_data import download_climate_data
from cyano.data.elevation_data import download_elevation_data
from cyano.data.features import generate_features
from cyano.data.satellite_data import download_satellite_data
from cyano.data.utils import load_sample_list, load_labels
from cyano.models.utils import EnsembledModel
from cyano.settings import MODEL_WEIGHTS_DIR, DEFAULT_TMP_FEATURES_DIR

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def train(
    labels_path: Path,
    model_config: Dict,
    model_save_dir: Path,
    prediction_col: str = "severity",
    features_dir: Path = DEFAULT_TMP_FEATURES_DIR,
):
    """Train a cyanobacteria prediction model

    Args:
        labels_path (Path): Path to a csv with columns for date,
            longitude, latitude, and severity
        model_config (Dict): Model hyperparameters
        model_save_dir (Path): Directory to save the trained model
        prediction_col (str, optional): Target column in the labels dataframe.
            Defaults to "severity".
        features_dir (Path, optional): Directory to save interim raw
            data from satellite, climate, and elevation sources. Defaults to
            DEFAULT_TMP_FEATURES_DIR
    """
    ## Load labels
    labels = load_labels(labels_path)
    logger.info(f"Loaded {labels.shape[0]:,} samples for training")

    ## Generate features for labeled samples
    samples = labels.drop(columns=[prediction_col])

    download_satellite_data(samples, features_dir=features_dir)
    download_climate_data(samples, features_dir=features_dir)
    download_elevation_data(samples, features_dir=features_dir)
    logger.success(f"Raw source data saved to {features_dir}")

    features = generate_features(samples, features_dir)
    logger.info(f"Generated {features.shape[1]:,} features for {features.shape[0]:,} samples")

    ## Instantiate model
    model = EnsembledModel(model_config)

    ## Train model and save
    logger.info(f"Training model with config: {model.config}")
    model.train(features, labels)

    logger.info(f"Saving model to {model_save_dir}")
    model.save(model_save_dir)


@app.command()
def predict(
    sample_list_path: Path,
    preds_save_path: Path,
    features_dir: Path = DEFAULT_TMP_FEATURES_DIR,
    model_weights_dir: Path = MODEL_WEIGHTS_DIR,
):
    """Load an existing cyanobacteria prediction model and generate
    severity level predictions for a set of samples.

    Args:
        sample_list_path (Path): Path to a csv with columns for date,
            longitude, and latitude
        preds_save_path (Path): Path to save the generated predictions
        features_dir (Path, optional): Directory to save interim raw
            data from satellite, climate, and elevation sources. Defaults to
            DEFAULT_TMP_FEATURES_DIR
        model_weights_dir (Path, optional): Directory with existing model
            weights and configuration to load. Defaults to MODEL_WEIGHTS_DIR
    """
    ## Load data
    samples = load_sample_list(sample_list_path)
    logger.info(f"Loaded {samples.shape[0]:,} samples for prediction")

    ## Query for from data sources and save
    download_satellite_data(samples, features_dir=features_dir)
    download_climate_data(samples, features_dir=features_dir)
    download_elevation_data(samples, features_dir=features_dir)
    logger.success(f"Raw source data saved to {features_dir}")

    ## Generate features
    features = generate_features(samples, features_dir)
    logger.info(f"Generated {features.shape[1]:,} features for {features.shape[0]:,} samples")

    ## Load model
    model = EnsembledModel.load_model(model_weights_dir)
    logger.info(f"Loaded model with config: {model.config}")

    ## Predict
    preds = model.predict(features)

    preds.to_csv(preds_save_path)
    logger.success(f"Predictions saved to {preds_save_path}")


if __name__ == "__main__":
    app()
