from enum import Enum
import sys
import tempfile

from loguru import logger
import pandas as pd
from pathlib import Path
from pyproj import Transformer
import shutil
import typer

from cyfi.pipeline import CyFiPipeline
from cyfi.evaluate import EvaluatePreds
from cyfi import visualize
from cyfi.version import __version__

app = typer.Typer(pretty_exceptions_show_locals=False)

DEFAULT_MODEL_PATH = str(Path(__file__).parent / "assets/model_v0.zip")


class CRS(str, Enum):
    EPSG_4326 = "EPSG:4326"
    EPSG_3857 = "EPSG:3857"


def verbose_callback(verbosity: int):
    """Set up logger with level based on --verbose count."""
    logger.remove()
    if verbosity >= 2:
        logger.add(sys.stderr, level="DEBUG")
    elif verbosity == 1:
        logger.add(sys.stderr, level="INFO")
    else:
        logger.add(sys.stderr, level="SUCCESS")


verbose_option = typer.Option(
    0,
    "--verbose",
    "-v",
    count=True,
    show_default=False,
    help="Increase the verbosity/log level. [-v = INFO, -vv = DEBUG]",
    callback=verbose_callback,
)


def version_callback(version: bool):
    """Print CyFi version to console."""
    if version:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        callback=version_callback,
        is_eager=True,
        help="Show CyFi version.",
    ),
):
    pass


@app.command()
def predict(
    samples_path: Path = typer.Argument(
        exists=True,
        help="Path to a csv of sample points with columns for date, longitude, and latitude. Latitude and longitude must be in coordinate reference system WGS-84 (EPSG:4326)",
    ),
    model_path: Path = typer.Option(
        default=None,
        exists=True,
        help="Path to the zipfile of a trained cyanobacteria prediction model. If no model is specified, the default model will be used",
    ),
    output_filename: Path = typer.Option(
        "preds.csv", "--output-filename", "-f", help="Name of the saved out predictions csv"
    ),
    output_directory: Path = typer.Option(
        Path.cwd(),
        "--output-directory",
        "-d",
        help="Directory to save prediction outputs. `output_filename` will be interpreted relative to `output_directory`",
    ),
    keep_features: bool = typer.Option(
        default=False, help="Whether to save sample features to `output_directory`"
    ),
    keep_metadata: bool = typer.Option(
        default=False, help="Whether to save Sentinel image metadata to `output_directory`"
    ),
    overwrite: bool = typer.Option(False, "--overwrite", "-o", help="Overwrite existing files"),
    verbose: int = verbose_option,
):
    """Estimate cyanobacteria density for a set of sample points saved at `samples_path`. By
    default, cyanobacteria estimates will be saved to `preds.csv` in the current directory.
    """
    output_path = output_directory / output_filename
    features_path = output_directory / "sample_features.csv"
    metadata_path = output_directory / "sentinel_metadata.csv"
    if not overwrite:
        if output_path.exists():
            raise FileExistsError(
                f"Not generating predictions because overwrite is False and {output_path} exists. To overwrite existing predictions, add `-o`."
            )
        if keep_features and features_path.exists():
            raise FileExistsError(
                f"Not generating predictions because overwrite is False and {features_path} exists. To overwrite existing features, add `-o`."
            )
        if keep_metadata and metadata_path.exists():
            raise FileExistsError(
                f"Not generating predictions because overwrite is False and {metadata_path} exists. To overwrite existing Sentinel metadata, add `-o`."
            )
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    pipeline = CyFiPipeline.from_disk(model_path)

    pipeline.run_prediction(samples_path, output_path)

    if keep_features:
        pipeline.predict_features.to_csv(features_path, index=True)
        logger.success(f"Features saved to {features_path}")
    if keep_metadata:
        shutil.copy(pipeline.cache_dir / "sentinel_metadata_test.csv", metadata_path)
        logger.success(f"Sentinel metadata saved to {metadata_path}")


@app.command()
def predict_point(
    latitude: float = typer.Option(..., "--lat", help="Latitude"),
    longitude: float = typer.Option(..., "--lon", help="Longitude"),
    date: str = typer.Option(
        None,
        "--date",
        "-dt",
        help="Date formatted as YYYY-MM-DD, e.g. 2023-09-20. If no date is specified, today's date will be used.",
    ),
    crs: CRS = typer.Option(
        "EPSG:4326",
        help="Coordinate reference system of the provided latitude and longitude.",
    ),
    verbose: int = verbose_option,
):
    """Estimate cyanobacteria density for a single location on a given date"""

    if date is None:
        date = pd.to_datetime("today").strftime("%Y-%m-%d")

    # check provided date is not in the future
    elif pd.to_datetime(date) > pd.to_datetime("today"):
        raise ValueError("Cannot predict on a date that is in the future.")

    transformer = Transformer.from_crs(crs_from=crs.value, crs_to="EPSG:4326")
    converted_latitude, converted_longitude = transformer.transform(latitude, longitude)

    samples = pd.DataFrame(
        {"date": [date], "latitude": [converted_latitude], "longitude": [converted_longitude]}
    )
    samples_path = Path(tempfile.gettempdir()) / "samples.csv"
    samples.to_csv(samples_path, index=False)

    pipeline = CyFiPipeline.from_disk(DEFAULT_MODEL_PATH)
    pipeline.run_prediction(samples_path, preds_path=None)

    # print out user-specified lat / lon
    pipeline.output_df["latitude"] = [latitude]
    pipeline.output_df["longitude"] = [longitude]

    # format as integer with comma for console
    pipeline.output_df["density_cells_per_ml"] = pipeline.output_df.density_cells_per_ml.map(
        "{:,.0f}".format
    )
    logger.success(f"Estimate generated:\n{pipeline.output_df.iloc[0].to_string()}")


@app.command()
def evaluate(
    y_pred_csv: Path = typer.Argument(
        exists=True,
        help="Path to a csv of sample points with columns for date, longitude, latitude, and predicted density. Latitude and longitude must be in coordinate reference system WGS-84 (EPSG:4326)",
    ),
    y_true_csv: Path = typer.Argument(
        exists=True,
        help="Path to a csv of sample points with columns for date, longitude, latitude, and actual density, with optional metadata columns",
    ),
    save_dir: Path = typer.Option(
        default=Path.cwd() / "metrics", help="Folder in which to save out metrics and plots."
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", "-o", help="Overwrite any existing files in `save_dir`"
    ),
    verbose: int = verbose_option,
):
    """Evaluate cyanobacteria estimates"""
    if not overwrite and save_dir.exists():
        logger.warning(
            f"Not running evaluation because overwrite is False and {save_dir} exists. To overwrite existing files, add `-o`"
        )
        return

    EvaluatePreds(
        y_pred_csv=y_pred_csv, y_true_csv=y_true_csv, save_dir=save_dir
    ).calculate_all_and_save()


# add CyFi explorer
app.command()(visualize.visualize)


if __name__ == "__main__":
    app()
