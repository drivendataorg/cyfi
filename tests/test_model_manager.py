import yaml

import lightgbm as lgb
import pandas as pd
from pathlib import Path

from cyano.cli import train_model, predict_model
from cyano.models.cyano_model import CyanoModel


def test_train_model(train_config, train_data):
    # Run train model and check that it returns a model
    trained_model = train_model(train_data, train_config)
    assert isinstance(trained_model, CyanoModel)

    # Check that experiment config is saved correctly
    saved_config_path = Path(trained_model.config.save_dir) / "config.yaml"
    assert saved_config_path.exists()
    with open(saved_config_path, "r") as fp:
        saved_config = yaml.safe_load(fp)
    assert "features_config" in saved_config.keys()

    # Check that LGB Booster is saved correctly
    saved_lgb_path = Path(trained_model.config.save_dir) / "lgb_model.txt"
    assert saved_lgb_path.exists()
    lgb_model = lgb.Booster(model_file=saved_lgb_path)
    assert isinstance(lgb_model, lgb.Booster)
    assert lgb_model.feature_name() == train_config.features_config.satellite_features


def test_predict_model(predict_data, predict_config):
    # Run predict and check that it returns a dataframe
    preds = predict_model(predict_data, predict_config, debug=True)
    assert preds.shape[0] == predict_data.shape[0]

    # Check saved preds
    assert Path(predict_config.save_path).exists()
    saved_preds = pd.read_csv(predict_config.save_path)
    assert saved_preds.shape[0] == predict_data.shape[0]
