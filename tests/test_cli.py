import math

import pandas as pd
from pathlib import Path
from pyproj import Transformer
from pytest_mock import mocker  # noqa: F401
from typer.testing import CliRunner

from cyfi.cli import app
from cyfi.data.utils import add_unique_identifier
from cyfi.pipeline import CyFiPipeline


ASSETS_DIR = Path(__file__).parent / "assets"

runner = CliRunner()


def test_cli_predict(tmp_path, predict_data_path, predict_data, local_model_path):
    ## Run CLI command
    result = runner.invoke(
        app,
        [
            "predict",
            str(predict_data_path),
            "--model-path",
            str(local_model_path),
            "--output-directory",
            str(tmp_path),
            "--keep-features",
        ],
    )
    assert result.exit_code == 0

    # Check that preds saved out
    assert (tmp_path / "preds.csv").exists()
    preds = pd.read_csv(tmp_path / "preds.csv")
    assert (preds.index == predict_data.index).all()
    assert Path(tmp_path / "sample_features.csv").exists()

    # Check that the missing / non missing values are expected
    missing_sample_mask = preds.sample_id == "e66ea0c31ba500d5d4ac4c610b8cf508"
    assert preds[~missing_sample_mask].severity.notna().all()
    assert preds[missing_sample_mask].severity.isna().all()

    # Check that log level is expected
    assert "SUCCESS" in result.stdout
    assert "INFO" not in result.stdout


def test_cli_predict_samples_path(tmp_path, local_model_path):
    ## Errors when samples_path is not provided
    result = runner.invoke(
        app,
        [
            "predict",
            "--model-path",
            str(local_model_path),
            "--output-directory",
            str(tmp_path),
            "--overwrite",
        ],
    )
    assert result.exit_code == 2
    assert "Missing argument" in result.output

    # Errors when samples_path does not exist
    result = runner.invoke(
        app,
        [
            "predict",
            "not_a_path",
            "--model-path",
            str(ASSETS_DIR / "experiment/model.zip"),
            "--output-directory",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 2
    assert "does not exist" in result.stdout


def test_cli_no_overwrite(tmp_path, train_data, train_data_path, local_model_path):
    # Check that existing files aren't overwritten without the --overwrite flag
    train_data.to_csv(tmp_path / "preds.csv")

    result = runner.invoke(
        app,
        [
            "predict",
            str(train_data_path),
            "--model-path",
            str(local_model_path),
            "--output-directory",
            str(tmp_path),
            "--keep-features",
        ],
    )
    assert result.exit_code == 1
    assert isinstance(result.exception, FileExistsError)
    assert "overwrite" in result.exception.args[0]


def test_cli_predict_verbosity(tmp_path, predict_data_path, local_model_path):
    ## Check that log level increases when specified
    result = runner.invoke(
        app,
        [
            "predict",
            str(predict_data_path),
            "--model-path",
            str(local_model_path),
            "--output-directory",
            str(tmp_path),
            "--overwrite",
            "-v",
        ],
    )
    assert result.exit_code == 0
    assert "INFO" in result.stdout
    assert "DEBUG" not in result.stdout


# mock prediction to just test CLI args
def pipeline_predict_mock(self, predict_csv, preds_path=None):
    self.output_df = pd.DataFrame(
        {
            "date": ["2021-05-17"],
            "latitude": ["36.05"],
            "longitude": ["-76.7"],
            "severity": ["moderate"],
            "density_cells_per_ml": 32078.0,
        }
    )


def test_cli_predict_point(mocker):  # noqa: F811
    mocker.patch("cyfi.cli.CyFiPipeline.run_prediction", pipeline_predict_mock)

    result = runner.invoke(
        app,
        [
            "predict-point",
            "-dt",
            "2021-05-17",
            "-lat",
            "36.05",
            "-lon",
            "-76.7",
        ],
    )
    assert result.exit_code == 0

    # try predicting in future
    result = runner.invoke(
        app,
        [
            "predict-point",
            "-dt",
            "2035-01-01",
            "-lat",
            "36.05",
            "-lon",
            "-76.7",
        ],
    )
    assert result.exit_code == 1
    assert "Cannot predict on a date that is in the future" in result.exception.__str__()


def test_cli_predict_point_crs(mocker, ensembled_model_path, tmp_path):  # noqa: F811
    # Test specifying a point in a different CRS
    mocker.patch("cyano.cli.DEFAULT_MODEL_PATH", ensembled_model_path)

    (lat, lon, date) = (37.7, -122.4, "2022-09-01")

    # Get expected prediction value
    samples = pd.DataFrame({"date": [date], "latitude": [lat], "longitude": [lon]})
    samples_path = tmp_path / "samples.csv"
    samples.to_csv(samples_path, index=False)

    pipeline = CyanoModelPipeline.from_disk(ensembled_model_path)
    pipeline.run_prediction(tmp_path / "samples.csv")
    expected_density = pipeline.output_df["density_cells_per_ml"].iloc[0]

    # Run CLI with different CRS
    use_crs = "EPSG:3857"
    transformer = Transformer.from_crs("EPSG:4326", use_crs)
    (new_lat, new_lon) = transformer.transform(lat, lon)

    result = runner.invoke(
        app,
        [
            "predict-point",
            "-dt",
            date,
            "-lat",
            str(new_lat),
            "-lon",
            str(new_lon),
            "--crs",
            use_crs,
        ],
    )
    assert result.exit_code == 0
    assert f"{expected_density:,.0f}" in result.stdout
    assert str(math.trunc(new_lat)) in result.stdout
    assert str(math.trunc(new_lon)) in result.stdout


def test_cli_evaluate(tmp_path, evaluate_data_path):
    df = pd.read_csv(evaluate_data_path)
    df = add_unique_identifier(df)
    data_path = tmp_path / "data.csv"
    df.to_csv(data_path, index=True)

    eval_dir = tmp_path / "evals"

    # Run CLI command
    result = runner.invoke(
        app,
        ["evaluate", str(data_path), str(data_path), "--save-dir", str(eval_dir)],
    )
    assert result.exit_code == 0

    # Check that appropriate files are in the eval_dir
    for file in [
        "actual_density_boxplot.png",
        "crosstab.png",
        "density_kde.png",
        "density_scatterplot.png",
        "results.json",
    ]:
        assert (eval_dir / file).exists()
