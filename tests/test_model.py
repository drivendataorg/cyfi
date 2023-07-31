import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from pathlib import Path
import pytest
from typer.testing import CliRunner

from cyano.cli import train_model, predict_model, app
from cyano.data.features import generate_features
from cyano.models.cyano_model import CyanoModel

runner = CliRunner()


def test_train_model(tmp_path_factory, train_config, train_data):
    train_config["cyano_model_train_config"] = {"trained_model_dir": str(tmp_path_factory.mktemp("model_dir"))}

    # Run train model and check that it returns a model
    trained_model = train_model(train_data, train_config)
    assert type(trained_model) == CyanoModel

    # Check that model config is saved correctly
    saved_config_path = Path(trained_model.config.trained_model_dir) / "run_config.json"
    assert saved_config_path.exists()
    with open(saved_config_path, "r") as fp:
        saved_config = json.load(fp)
    assert "trained_model_dir" in saved_config.keys()

    # Check that LGB Booster is saved correctly
    saved_lgb_path = Path(trained_model.config.trained_model_dir) / "lgb_model.txt"
    assert saved_lgb_path.exists()
    lgb_model = lgb.Booster(model_file=saved_lgb_path)
    assert type(lgb_model) == lgb.Booster
    assert lgb_model.feature_name() == train_config["satellite_features"]


def test_predict_model(tmp_path_factory, predict_data, predict_config):
    # Run predict and check that it returns a dataframe
    preds_path = tmp_path_factory.mktemp("test_predict") / "preds.csv"
    predict_config["preds_save_path"] = str(preds_path)

    preds = predict_model(predict_data, predict_config, debug=True)
    assert preds.shape[0] == predict_data.shape[0]

    # Check saved preds
    assert preds_path.exists()
    saved_preds = pd.read_csv(preds_path)
    assert saved_preds.shape[0] == predict_data.shape[0]


# def test_cli_train(tmp_path_factory):
#     tmp_cli_train_dir = tmp_path_factory.mktemp("test_cli_train")

#     # Write out config with model save dir in tmp dir
#     config = TRAIN_CONFIG.copy()
#     config["cyano_model_config"] = {"trained_model_dir": str(tmp_cli_train_dir / "trained_model")}
#     config_path = tmp_cli_train_dir / "config.json"
#     with open(config_path, "w") as fp:
#         json.dump(config, fp)

#     # Run cli command
#     result = runner.invoke(
#         app,
#         ["train", str(ASSETS_DIR / "train_data.csv"), str(config_path), "--debug"],
#     )
#     assert result.exit_code == 0

#     # Check that model config saved out
#     saved_config_path = Path(config["cyano_model_config"]["trained_model_dir"]) / "run_config.json"
#     assert saved_config_path.exists()

#     # Check that LGB Booster saved out
#     saved_lgb_path = Path(config["cyano_model_config"]["trained_model_dir"]) / "lgb_model.txt"
#     assert saved_lgb_path.exists()


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


# def test_known_features(train_data: pd.DataFrame):
#     config = PREDICT_CONFIG.copy()

#     # Generate features based on saved imagery
#     config["cache_dir"] = str(ASSETS_DIR / "feature_cache")
#     config = TrainConfig(**config)
#     features = generate_features(train_data.set_index("uid"), config)

#     # Check that generated stats match known imagery stats
#     assert np.isclose(features.loc["ofhd", "B02_mean"], 161.532712)
#     assert np.isclose(features.loc["ofhd", "B02_min"], 50)
#     assert np.isclose(features.loc["ofhd", "B02_max"], 1182)
