import sys
from zipfile import ZipFile

from cloudpathlib import AnyPath
from dotenv import load_dotenv, find_dotenv
import lightgbm as lgb
from loguru import logger
from pandas_path import path  # noqa
from pathlib import Path
import typer

from cyano.experiment.experiment import ExperimentConfig
from cyano.pipeline import CyanoModelPipeline
from cyano.evaluate import EvaluatePreds

import pandas as pd
from cyano.data.elevation_data import download_elevation_data
from cyano.config import FeaturesConfig
from cyano.data.utils import add_unique_identifier
from cyano.settings import REPO_ROOT

app = typer.Typer(pretty_exceptions_show_locals=False)

load_dotenv(find_dotenv())

# Set logger to only log info or higher
logger.remove()
logger.add(sys.stderr, level="INFO")


@app.command()
def experiment(
    config_path: Path = typer.Argument(exists=True, help="Path to an experiment configuration")
):
    """Run an experiment"""
    config = ExperimentConfig.from_file(config_path)
    logger.add(config.save_dir / "experiment.log", level="DEBUG")
    config.run_experiment()


@app.command()
def predict(
    samples_path: Path = typer.Argument(
        exists=True, help="Path to a csv of samples with columns for date, longitude, and latitude"
    ),
    model_zip: Path = typer.Argument(exists=True, help="Path to a trained model zip"),
    output_path: Path = typer.Option(
        default="preds.csv", help="Destination to save predictions csv"
    ),
):
    """Load an existing cyanobacteria prediction model and generate
    severity level predictions for a set of samples.
    """
    samples_path = AnyPath(samples_path)

    pipeline = CyanoModelPipeline.from_disk(model_zip)
    pipeline.run_prediction(samples_path, output_path)


@app.command()
def evaluate(
    y_pred_csv: str = typer.Argument(
        help="Path to a csv of samples with columns for date, longitude, latitude, and predicted severity",
    ),
    y_true_csv: str = typer.Argument(
        help="Path to a csv of samples with columns for date, longitude, latitude, and actual severity, with optional metadata columns",
    ),
    model_path: Path = typer.Argument(exists=True, help="Path to trained LGB model"),
    save_dir: Path = typer.Option(
        default=Path.cwd() / "metrics", help="Folder in which to save out metrics and plots."
    ),
):
    y_pred_csv = AnyPath(y_pred_csv)
    y_true_csv = AnyPath(y_true_csv)

    # Load model from model zipfile
    archive = ZipFile(model_path, "r")
    model = lgb.Booster(model_str=archive.read("lgb_model.txt").decode())

    EvaluatePreds(
        y_pred_csv=y_pred_csv, y_true_csv=y_true_csv, save_dir=save_dir, model=model
    ).calculate_all_and_save()


@app.command()
def download_elevation(
    split: str,
    data_dir: str = "s3://drivendata-competition-nasa-cyanobacteria/experiments/splits/competition",
    meters_window: int = 1000,
    cache_dir: str = str(REPO_ROOT.parent / "experiments/cache"),
):
    cache_dir = AnyPath(cache_dir)
    logger.add(cache_dir / "elevation_download.log", level="DEBUG")

    config = FeaturesConfig(elevation_feature_meter_window=meters_window)
    logger.info(
        f"Downloading elevation data with window {config.elevation_feature_meter_window:,}m to {cache_dir}"
    )

    df = pd.read_csv(AnyPath(data_dir) / f"{split}.csv")
    df = add_unique_identifier(df)[["longitude", "latitude", "date"]]
    logger.info(f"Loaded {df.shape[0]:,} {split} samples")

    df["elev_exists"] = cache_dir / "elevation_1000" / df.index.path.with_suffix(".json")
    df["elev_exists"] = df.elev_exists.path.exists()
    logger.info(f"Elevation data already exists for {df.elev_exists.sum()} samples")

    df = df[~df.elev_exists]
    download_elevation_data(df, config, cache_dir)


if __name__ == "__main__":
    app()
