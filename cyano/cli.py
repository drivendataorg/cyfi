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


def convert_anypath(var_name: str, filepath: str) -> AnyPath:
    """Convert a string to AnyPath and check that the path exists

    Args:
        var_name (str): Name of the filepath variable for logging errors
        filepath (str): File path
    """
    filepath = AnyPath(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Invalid value for {var_name}: {filepath} does not exist.")

    return filepath


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
    model_path: str = typer.Option(
        default=str(REPO_ROOT / "cyano/assets/model.zip"),
        help="Path to the zipfile of a trained LGB model",
    ),
    output_path: Path = typer.Option(
        default="preds.csv", help="Destination to save predictions csv"
    ),
    cache_dir: Path = typer.Option(default=None, help="Cache directory to save raw feature data"),
):
    """Generate cyanobacteria predictions for a set of samples using an existing cyanobacteria prediction model"""
    # Convert strings to AnyPath and check that they exist
    samples_path = convert_anypath("SAMPLES_PATH", samples_path)
    model_path = convert_anypath("MODEL_PATH", model_path)

    logger.info(f"Loading trained model from {model_path}")
    pipeline = CyanoModelPipeline.from_disk(model_path, cache_dir=cache_dir)
    pipeline.run_prediction(samples_path, output_path)


@app.command()
def evaluate(
    y_pred_csv: str = typer.Argument(
        help="Path to a csv of samples with columns for date, longitude, latitude, and predicted density",
    ),
    y_true_csv: str = typer.Argument(
        help="Path to a csv of samples with columns for date, longitude, latitude, and actual density, with optional metadata columns",
    ),
    model_path: str = typer.Option(
        default=str(REPO_ROOT / "cyano/assets/model.zip"),
        help="Path to the zipfile of a trained LGB model",
    ),
    save_dir: Path = typer.Option(
        default=Path.cwd() / "metrics", help="Folder in which to save out metrics and plots."
    ),
):
    """Evaluate cyanobacteria model predictions"""
    # Convert strings to AnyPath and check that they exist
    y_pred_csv = convert_anypath("Y_PRED_CSV", y_pred_csv)
    y_true_csv = convert_anypath("Y_TRUE_CSV", y_true_csv)
    model_path = convert_anypath("MODEL_PATH", model_path)

    EvaluatePreds(
        y_pred_csv=y_pred_csv, y_true_csv=y_true_csv, save_dir=save_dir, model_path=model_path
    ).calculate_all_and_save()


if __name__ == "__main__":
    app()
