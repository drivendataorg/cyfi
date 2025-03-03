from pathlib import Path
from pydantic import ValidationError
import pytest
from typer.testing import CliRunner
import yaml

from cyfi.config import FeaturesConfig
from cyfi.experiment import app, ExperimentConfig

ASSETS_DIR = Path(__file__).parent / "assets"

runner = CliRunner()


def test_experiment_config(train_data_path):
    config = ExperimentConfig(
        train_csv=train_data_path,
        predict_csv=train_data_path,
        features_config=FeaturesConfig(n_sentinel_items=10),
    )
    assert config.features_config.n_sentinel_items == 10

    # Errors with extra field
    with pytest.raises(ValidationError):
        ExperimentConfig(
            train_csv=train_data_path,
            predict_csv=train_data_path,
            extra_field="surprise_extra_field",
        )


def test_cli_experiment(experiment_config_path, tmp_path):
    # use tmp_path as the save location
    config = ExperimentConfig.from_file(experiment_config_path)
    config.save_dir = tmp_path
    with (config.save_dir / "config_artifact.yaml").open("w") as fp:
        yaml.dump(config.model_dump(), fp)

    # Run CLI command
    result = runner.invoke(
        app,
        [str(experiment_config_path)],
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
