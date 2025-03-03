import shutil
import signal
import subprocess
import time
import pandas as pd
from pathlib import Path
from pyproj import Transformer
from pytest_mock import mocker  # noqa: F401
from typer.testing import CliRunner

from cyfi.cli import app
from cyfi.data.utils import add_unique_identifier


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

    # Check that the missing values are expected
    assert pd.isna(preds.set_index("sample_id").loc["e66ea0c31ba500d5d4ac4c610b8cf508"].severity)
    assert preds.severity.isna().sum() == 1

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
            "--lat",
            "36.05",
            "--lon",
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
            "--lat",
            "36.05",
            "--lon",
            "-76.7",
        ],
    )
    assert result.exit_code == 1
    assert "Cannot predict on a date that is in the future" in result.exception.__str__()


def test_cli_predict_point_crs(mocker, local_model_path):  # noqa: F811
    mocker.patch("cyfi.cli.DEFAULT_MODEL_PATH", local_model_path)

    transformer = Transformer.from_crs(crs_from="EPSG:4326", crs_to="EPSG:3857")
    lat, lon = transformer.transform(36.05, -76.7)

    inputs = [
        "predict-point",
        "-dt",
        "2023-01-01",
        "--lat",
        lat,
        "--lon",
        lon,
    ]
    # try predicting in a different CRS
    result = runner.invoke(
        app,
        inputs + ["--crs", "EPSG:3857"],
    )
    assert result.exit_code == 0

    # check original location printed out
    assert str(round(lat, 2)) in result.stdout

    # try predicting with invalid CRS
    result = runner.invoke(
        app,
        inputs + ["--crs", "EPSG:10"],
    )
    assert result.exit_code == 2
    assert "Invalid value for '--crs" in result.stdout


def test_graceful_exit_when_no_satellite_data():
    # use location and date in SF where it was cloudy all month and there is no valid imagery
    result = runner.invoke(
        app,
        [
            "predict-point",
            "--date",
            "2024-01-04",
            "--lat",
            "37.753",
            "--lon",
            "-122.364",
        ],
    )
    assert result.exit_code == 1
    assert (
        "Relevant satellite data is not available for any of the provided sample points. Please try a different location or date"
        in result.stdout
    )


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


def test_python_m_execution():
    result = subprocess.run(
        ["python", "-m", "cyfi", "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    assert result.returncode == 0
    assert "Usage: python -m cyfi" in result.stdout


def test_cyfi_explorer_launches(tmp_path):
    shutil.copy(ASSETS_DIR / "experiment" / "preds.csv", tmp_path / "preds.csv")
    shutil.copy(
        ASSETS_DIR / "experiment" / "sentinel_metadata_test.csv",
        tmp_path / "sentinel_metadata.csv",
    )

    proc = subprocess.Popen(
        ["cyfi", "visualize", str(tmp_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    time.sleep(10)
    proc.send_signal(signal.SIGINT)
    stdout, stderr = proc.communicate()
    assert "Running on" in stdout
