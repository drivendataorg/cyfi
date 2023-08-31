import sys

from cloudpathlib import AnyPath
from dotenv import load_dotenv, find_dotenv
from loguru import logger
from pathlib import Path
import typer

from cyano.experiment.experiment import ExperimentConfig
from cyano.pipeline import CyanoModelPipeline
from cyano.evaluate import EvaluatePreds

app = typer.Typer(pretty_exceptions_show_locals=False)

load_dotenv(find_dotenv())

REPO_ROOT = Path(__file__).parents[1].resolve()

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
    samples_path: str = typer.Argument(
        help="Path to a csv of samples with columns for date, longitude, and latitude"
    ),
    model_zip: str = typer.Option(
        default=str(REPO_ROOT / "cyano/assets/model.zip"),
        help="Path to a trained model zip",
    ),
    output_path: Path = typer.Option(
        default="preds.csv", help="Destination to save predictions csv"
    ),
    cache_dir: Path = typer.Option(default=None, help="Cache directory to save raw feature data"),
):
    """Load an existing cyanobacteria prediction model and generate
    severity level predictions for a set of samples.
    """
    samples_path = AnyPath(samples_path)
    model_zip = AnyPath(model_zip)

    if not samples_path.exists():
        raise FileNotFoundError(
            f"Invalid valid for 'SAMPLES_PATH': {samples_path} does not exist."
        )
    if not model_zip.exists():
        raise FileNotFoundError(f"Invalid value for 'MODEL_ZIP': {model_zip} does not exist.")

    logger.info(f"Loading trained model from {model_zip}")
    pipeline = CyanoModelPipeline.from_disk(model_zip, cache_dir=cache_dir)
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

    EvaluatePreds(
        y_pred_csv=y_pred_csv, y_true_csv=y_true_csv, save_dir=save_dir, model_path=model_path
    ).calculate_all_and_save()


if __name__ == "__main__":
    app()
