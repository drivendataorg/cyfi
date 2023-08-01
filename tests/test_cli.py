import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from pathlib import Path
from typer.testing import CliRunner

from cyano.cli import app

ASSETS_DIR = Path(__file__).parent / "assets"

runner = CliRunner()


def test_cli_train(tmp_path, train_config):
    # Write out config with model save dir in tmp dir
    config_path = tmp_path / "train_config.json"
    with open(config_path, "w") as fp:
        json.dump(train_config.model_dump(), fp)

    # Run CLI command
    result = runner.invoke(
        app,
        ["train", str(ASSETS_DIR / "train_data.csv"), str(config_path)],
    )
    assert result.exit_code == 0

    # Check that model config saved out
    assert (Path(train_config.tree_model_config.save_dir) / "run_config.json").exists()

    # Check that LGB Booster saved out
    assert (Path(train_config.tree_model_config.save_dir) / "lgb_model.txt").exists()


# def test_cli_predict(tmp_path_factory, train_data: pd.DataFrame):
#     tmp_cli_predict_dir = tmp_path_factory.mktemp("test_cli_predict")

#     # Write out config into tmp dir
#     config = PREDICT_CONFIG.copy()
#     preds_path = tmp_cli_predict_dir / "preds.csv"
#     config["preds_save_path"] = str(preds_path)
#     config_path = tmp_cli_predict_dir / "config.json"
#     with open(config_path, "w") as fp:
#         json.dump(config, fp)

#     # Run cli command
#     result = runner.invoke(
#         app,
#         [
#             "predict",
#             str(ASSETS_DIR / "train_data.csv"),
#             str(config_path),
#             "--debug",
#         ],
#     )
#     assert result.exit_code == 0

#     # Check that preds saved out
#     assert preds_path.exists()
#     preds = pd.read_csv(preds_path)
#     assert preds.shape[0] == train_data.shape[0]
