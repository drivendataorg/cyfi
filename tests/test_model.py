import json
import tempfile

import lightgbm as lgb
import numpy as np
import pandas as pd
from pathlib import Path
import pytest
from typer.testing import CliRunner

from cyano.cli import train_model, predict_model, app
from cyano.config import ExperimentConfig
from cyano.data.features import generate_features
from cyano.models.cyano_model import CyanoModel

runner = CliRunner()

ASSETS_DIR = Path(__file__).parent / "assets"

TRAIN_CONFIG = {
    "use_sentinel_bands": ["B02"],
    "image_feature_meter_window": 500,
    "satellite_features": ["B02_mean", "B02_min", "B02_max"],
}

PREDICT_CONFIG = TRAIN_CONFIG.copy()
PREDICT_CONFIG["trained_model_dir"] = str(ASSETS_DIR / "trained_model")


@pytest.fixture
def train_data() -> pd.DataFrame:
    return pd.read_csv(ASSETS_DIR / "train_data.csv")


def test_train_model(train_data: pd.DataFrame):
    with tempfile.TemporaryDirectory() as model_dir:
        config = TRAIN_CONFIG.copy()
        config["trained_model_dir"] = model_dir

        # Run train model and check that it returns a model
        trained_model = train_model(train_data, config, debug=True)
        assert type(trained_model) == CyanoModel

        # Check that model config is saved correctly
        saved_config_path = Path(trained_model.config.trained_model_dir) / "run_config.json"
        assert saved_config_path.exists()
        with open(saved_config_path, "r") as fp:
            saved_config = json.load(fp)
        assert "satellite_features" in saved_config.keys()
        assert "cache_dir" in saved_config.keys()

        # Check that LGB Booster is saved correctly
        saved_lgb_path = Path(trained_model.config.trained_model_dir) / "lgb_model.txt"
        assert saved_lgb_path.exists()
        lgb_model = lgb.Booster(model_file=saved_lgb_path)
        assert type(lgb_model) == lgb.Booster
        assert lgb_model.feature_name() == config["satellite_features"]


def test_predict_model(train_data: pd.DataFrame):
    # Run predict and check that it returns a dataframe
    with tempfile.TemporaryDirectory() as preds_dir:
        preds_path = Path(preds_dir) / "preds.csv"
        preds = predict_model(train_data, preds_path, PREDICT_CONFIG, debug=True)
        assert preds.shape[0] == train_data.shape[0]

        # Check saved preds
        assert preds_path.exists()
        saved_preds = pd.read_csv(preds_path)
        assert saved_preds.shape[0] == train_data.shape[0]


def test_cli_train():
    with tempfile.TemporaryDirectory() as tmp_cli_train_dir:
        # Write out config with model save dir in tmp dir
        config = TRAIN_CONFIG.copy()
        config_path = f"{tmp_cli_train_dir}/config.json"
        config["trained_model_dir"] = f"{tmp_cli_train_dir}/trained_model"
        with open(config_path, "w") as fp:
            json.dump(config, fp)

        # Run cli command
        result = runner.invoke(
            app,
            ["train", str(ASSETS_DIR / "train_data.csv"), config_path, "--debug"],
        )
        assert result.exit_code == 0

        # Check that model config saved out
        saved_config_path = Path(config["trained_model_dir"]) / "run_config.json"
        assert saved_config_path.exists()

        # Check that LGB Booster saved out
        saved_lgb_path = Path(config["trained_model_dir"]) / "lgb_model.txt"
        assert saved_lgb_path.exists()


def test_cli_predict(train_data: pd.DataFrame):
    with tempfile.TemporaryDirectory() as tmp_cli_predict_dir:
        # Write out config into tmp dir
        config = PREDICT_CONFIG.copy()
        config_path = f"{tmp_cli_predict_dir}/config.json"
        with open(config_path, "w") as fp:
            json.dump(config, fp)

        # Run cli command
        preds_path = f"{tmp_cli_predict_dir}/preds.csv"
        result = runner.invoke(
            app,
            [
                "predict",
                str(ASSETS_DIR / "train_data.csv"),
                config_path,
                preds_path,
                "--debug",
            ],
        )
        assert result.exit_code == 0

        # Check that preds saved out
        assert Path(preds_path).exists()
        preds = pd.read_csv(preds_path)
        assert preds.shape[0] == train_data.shape[0]


def test_known_features(train_data: pd.DataFrame):
    config = PREDICT_CONFIG.copy()

    # Generate features based on saved imagery
    config["cache_dir"] = str(ASSETS_DIR / "feature_cache")
    config = ExperimentConfig(**config)
    features = generate_features(train_data.set_index("uid"), config)

    # Check that generated stats match known imagery stats
    assert np.isclose(features.loc["ofhd", "B02_mean"], 161.532712)
    assert np.isclose(features.loc["ofhd", "B02_min"], 50)
    assert np.isclose(features.loc["ofhd", "B02_max"], 1182)
