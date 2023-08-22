import yaml

import pandas as pd
from pathlib import Path
from typer.testing import CliRunner

from cyano.cli import app
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
    preds_path = tmp_path / "preds.csv"

    # Run CLI command
    result = runner.invoke(
        app,
        [
            "predict",
            str(predict_data_path),
            str(ASSETS_DIR / "experiment/model.zip"),
            "--output-path",
            str(preds_path),
        ],
    )
    assert result.exit_code == 0

    # Check that preds saved out
    assert preds_path.exists()
    preds = pd.read_csv(preds_path)
    assert (preds.index == predict_data.index).all()

    # Check that the missing / non missing values are expected
    missing_sample_mask = preds.sample_id == "e66ea0c31ba500d5d4ac4c610b8cf508"
    assert preds[~missing_sample_mask].severity.notna().all()
    assert preds[missing_sample_mask].severity.isna().all()
