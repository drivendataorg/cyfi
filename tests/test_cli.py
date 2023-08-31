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

    config = ExperimentConfig.from_file(experiment_config_path)

    # Check that artifact config, model zip, and predictions got saved out
    for file in ["config_artifact.yaml", "model.zip", "preds.csv"]:
        assert (Path(config.save_dir) / file).exists()

    # Check the appropriate files are in the metrics directory
    for file in [
        "actual_density_boxplot.png",
        "crosstab.png",
        "density_kde.png",
        "density_scatterplot.png",
        "feature_importance_model_0.csv",
        "results.json",
    ]:
        assert (Path(config.save_dir) / "metrics" / file).exists()


def test_cli_predict(tmp_path, predict_data_path, predict_data):
    preds_path = tmp_path / "preds.csv"

    # Run CLI command
    result = runner.invoke(
        app,
        [
            "predict",
            str(predict_data_path),
            "--model-path",
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


def test_cli_predict_invalid_files(tmp_path):
    # Raises an error when samples_path does not exist
    result = runner.invoke(
        app,
        [
            "predict",
            "not_a_path",
            "--model-path",
            str(ASSETS_DIR / "experiment/model.zip"),
            "--output-path",
            str(tmp_path / "preds.csv"),
        ],
    )
    assert result.exit_code == 1
    assert isinstance(result.exception, FileNotFoundError)


def test_cli_evaluate(tmp_path, data_with_sample_id_path):
    # Check that evaluate runs with the default model path
    eval_dir = tmp_path / "evals"

    # Run CLI command
    result = runner.invoke(
        app,
        [
            "evaluate",
            str(data_with_sample_id_path),
            str(data_with_sample_id_path),
            "--save-dir",
            str(eval_dir),
        ],
    )
    assert result.exit_code == 0

    # Check that appropriate files are in the eval_dir
    for file in [
        "actual_density_boxplot.png",
        "crosstab.png",
        "density_kde.png",
        "density_scatterplot.png",
        "feature_importance_model_0.csv",
        "results.json",
    ]:
        assert (eval_dir / file).exists()
