import json

import pandas as pd
from pathlib import Path
from typer.testing import CliRunner

from cyano.cli import app

ASSETS_DIR = Path(__file__).parent / "assets"

runner = CliRunner()


def test_cli_train(tmp_path, train_data_path, train_config):
    # Write out config to tmp dir
    config_path = tmp_path / "train_config.json"
    with open(config_path, "w") as fp:
        json.dump(train_config.model_dump(), fp)

    # Run CLI command
    result = runner.invoke(
        app,
        ["train", str(train_data_path), str(config_path)],
    )
    assert result.exit_code == 0

    # Check that experiment config saved out
    assert (Path(train_config.tree_model_config.save_dir) / "config.json").exists()

    # Check that LGB Booster saved out
    assert (Path(train_config.tree_model_config.save_dir) / "lgb_model.txt").exists()


def test_cli_predict(tmp_path, predict_data_path, predict_data, predict_config):
    # Write out config to tmp dir
    config_path = tmp_path / "predict_config.json"
    with open(config_path, "w") as fp:
        json.dump(predict_config.model_dump(), fp)

    # Run CLI command
    result = runner.invoke(
        app,
        ["predict", str(predict_data_path), str(config_path)],
    )
    assert result.exit_code == 0

    # Check that preds saved out
    assert Path(predict_config.save_path).exists()
    preds = pd.read_csv(predict_config.save_path)
    assert len(preds) == len(predict_data)
