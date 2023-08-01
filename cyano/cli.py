import yaml

import pandas as pd
from pathlib import Path
import typer

from cyano.config import TrainConfig, PredictConfig
from cyano.model_manager import train_model, predict_model

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def train(
    labels_path: Path = typer.Argument(
        exists=True, help="Path to a csv with columns for date, longitude, latitude, and severity"
    ),
    config_path: Path = typer.Argument(exists=True, help="Path to a train configuration"),
):
    """Train a cyanobacteria prediction model based on the labels in labels_path
    and the config file saved at config_path. The trained model and full experiment
    configuration will be saved to the "trained_model_dir" specified in the config
    """
    with open(config_path, "r") as fp:
        config_dict = yaml.safe_load(fp)
        config = TrainConfig(**config_dict)

    labels = pd.read_csv(labels_path)

    train_model(labels, config)


@app.command()
def predict(
    samples_path: Path = typer.Argument(
        exists=True, help="Path to a csv of samples with columns for date, longitude, and latitude"
    ),
    config_path: Path = typer.Argument(exists=True, help="Path to an experiment configuration"),
    # preds_path: Path = typer.Argument(help="Destination to save predictions csv"),
):
    """Load an existing cyanobacteria prediction model from trained_model_dir and generate
    severity level predictions for a set of samples."""
    with open(config_path, "r") as fp:
        config_dict = yaml.safe_load(fp)
        config = PredictConfig(**config_dict)

    samples = pd.read_csv(samples_path)

    predict_model(samples, config=config)


if __name__ == "__main__":
    app()
