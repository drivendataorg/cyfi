import yaml

import pandas as pd
from pandas_path import path  # noqa
from pathlib import Path
from typer.testing import CliRunner
from zipfile import ZipFile

from cyano.cli import app
from cyano.config import FeaturesConfig
from cyano.experiment.experiment import ExperimentConfig

ASSETS_DIR = Path(__file__).parent / "assets"

runner = CliRunner()


def test_cli_experiment(experiment_config_path):
    # Run CLI command
    result = runner.invoke(
        app,
        ["experiment", str(experiment_config_path)],
    )
    assert result.exit_code == 0

    with experiment_config_path.open("r") as f:
        config = ExperimentConfig(**yaml.safe_load(f))

    # Check that artifact config, model zip, and predictions got saved out
    assert (Path(config.save_dir) / "config_artifact.yaml").exists()
    assert (Path(config.save_dir) / "model.zip").exists()
    assert (Path(config.save_dir) / "preds.csv").exists()


def test_cli_predict(tmp_path, predict_data_path, predict_data):
    model_path = ASSETS_DIR / "experiment/model.zip"
    preds_path = tmp_path / "preds.csv"
    cache_dir = tmp_path / "cache"

    # Run CLI command
    result = runner.invoke(
        app,
        [
            "predict",
            str(predict_data_path),
            str(model_path),
            "--output-path",
            str(preds_path),
            "--cache-dir",
            str(cache_dir),
        ],
    )
    assert result.exit_code == 0

    # Check that preds saved out
    assert preds_path.exists()
    preds = pd.read_csv(preds_path)
    assert (preds.index == predict_data.index).all()

    # Load features config to get sentinel meter buffer
    archive = ZipFile(model_path, "r")
    features_config = FeaturesConfig(**yaml.safe_load(archive.read("config.yaml")))

    # Determine which samples have satellite imagery
    preds["sat_dir"] = (
        cache_dir / f"sentinel_{features_config.image_feature_meter_window}" / preds.sample_id.path
    )
    # Make sure those samples have predictions
    assert preds[preds.sat_dir.path.exists()].severity.notna().any()
