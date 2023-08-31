import sys

from dotenv import load_dotenv, find_dotenv
from loguru import logger
from pathlib import Path
import typer

from cyano.pipeline import CyanoModelPipeline
from cyano.evaluate import EvaluatePreds

app = typer.Typer(pretty_exceptions_show_locals=False)

load_dotenv(find_dotenv())

DEFAULT_MODEL_PATH = str(Path(__file__).parent / "assets/model_v0.zip")

# Set logger to only log info or higher
logger.remove()
logger.add(sys.stderr, level="INFO")


@app.command()
def predict(
    samples_path: Path = typer.Argument(
        exists=True, help="Path to a csv of samples with columns for date, longitude, and latitude"
    ),
    model_path: Path = typer.Option(
        default=None,
        exists=True,
        help="Path to the zipfile of a trained cyanobacteria prediction model. If no model is specified, the default model will be used",
    ),
    output_file: Path = typer.Option(
        default="preds.csv", help="Destination to save predictions csv"
    ),
    output_directory: Path = typer.Option(
        default=".",
        help="Directory to save prediction outputs. `output_file` will be interpreted relative to `output_directory`",
    ),
    keep_features: bool = typer.Option(
        default=False, help="Whether to save sample features to `output_directory`"
    ),
    overwrite: bool = typer.Option(default=False, help="Overwrite existing files"),
):
    """Generate cyanobacteria predictions for a set of samples saved at `samples_path`. By default,
    predictions will be saved to `preds.csv` in the current directory.
    """
    output_path = output_directory / output_file
    features_path = output_directory / "sample_features.csv"
    if not overwrite:
        if output_path.exists():
            raise FileExistsError(
                f"Not generating predictions because overwrite is False and {output_path} exists. To overwrite existing predictions, add `--overwrite`."
            )
        if keep_features and features_path.exists():
            raise FileExistsError(
                f"Not generating predictions because overwrite is False and {features_path} exists. To overwrite existing features, add `--overwrite`."
            )

    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    pipeline = CyanoModelPipeline.from_disk(model_path)

    pipeline.run_prediction(samples_path, output_path)

    if keep_features:
        pipeline.predict_features.to_csv(features_path, index=True)
        logger.success(f"Features saved to {features_path}")


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
    overwrite: bool = typer.Option(
        default=False, help="Overwrite any existing files in `save_dir`"
    ),
):
    """Evaluate cyanobacteria model predictions"""
    if not overwrite and save_dir.exists():
        logger.warning(
            f"Not running evaluation because overwrite is False and {save_dir} exists. To overwrite existing files, add `--overwrite`"
        )
        return

    EvaluatePreds(
        y_pred_csv=y_pred_csv, y_true_csv=y_true_csv, save_dir=save_dir
    ).calculate_all_and_save()


if __name__ == "__main__":
    app()
