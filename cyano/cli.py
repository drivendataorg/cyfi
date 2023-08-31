import sys

from dotenv import load_dotenv, find_dotenv
from loguru import logger
from pathlib import Path
import typer

from cyano.experiment.experiment import ExperimentConfig
from cyano.pipeline import CyanoModelPipeline
from cyano.evaluate import EvaluatePreds

app = typer.Typer(pretty_exceptions_show_locals=False)

load_dotenv(find_dotenv())

DEFAULT_MODEL_PATH = str(Path(__file__).parent / "assets/model_v0.zip")

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
    model_path: Path = typer.Option(
        default=None,
        exists=True,
        help="Path to the zipfile of a trained LGB model. If no model is specified, the default model will be used",
    ),
    output_path: Path = typer.Option(
        default="preds.csv", help="Destination to save predictions csv"
    ),
):
    """Generate cyanobacteria predictions for a set of samples using an existing cyanobacteria prediction model"""
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH

    pipeline = CyanoModelPipeline.from_disk(model_path)
    pipeline.run_prediction(samples_path, output_path)


@app.command()
def evaluate(
    y_pred_csv: Path = typer.Argument(
        exists=True,
        help="Path to a csv of samples with columns for date, longitude, latitude, and predicted density",
    ),
    y_true_csv: Path = typer.Argument(
        exists=True,
        help="Path to a csv of samples with columns for date, longitude, latitude, and actual density, with optional metadata columns",
    ),
    save_dir: Path = typer.Option(
        default=Path.cwd() / "metrics", help="Folder in which to save out metrics and plots."
    ),
):
    """Evaluate cyanobacteria model predictions"""
    EvaluatePreds(
        y_pred_csv=y_pred_csv, y_true_csv=y_true_csv, save_dir=save_dir
    ).calculate_all_and_save()


if __name__ == "__main__":
    app()
