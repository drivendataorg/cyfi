import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from pathlib import Path

from cyano.cli import train_model, predict_model
from cyano.models.cyano_model import CyanoModel


def test_train_model(train_config, train_data):
    import pdb; pdb.set_trace()
    # Run train model and check that it returns a model
    trained_model = train_model(train_data, train_config)
    assert type(trained_model) == CyanoModel

    # Check that model config is saved correctly
    saved_config_path = Path(trained_model.config.save_dir) / "run_config.json"
    assert saved_config_path.exists()
    with open(saved_config_path, "r") as fp:
        saved_config = json.load(fp)
    assert "save_dir" in saved_config.keys()

    # Check that LGB Booster is saved correctly
    saved_lgb_path = Path(trained_model.config.save_dir) / "lgb_model.txt"
    assert saved_lgb_path.exists()
    lgb_model = lgb.Booster(model_file=saved_lgb_path)
    assert type(lgb_model) == lgb.Booster
    assert lgb_model.feature_name() == train_config.features_config.satellite_features


# def test_predict_model(tmp_path_factory, predict_data, predict_config):
#     # Run predict and check that it returns a dataframe
#     preds_path = tmp_path_factory.mktemp("test_predict") / "preds.csv"
#     predict_config["preds_save_path"] = str(preds_path)

#     preds = predict_model(predict_data, predict_config, debug=True)
#     assert preds.shape[0] == predict_data.shape[0]

#     # Check saved preds
#     assert preds_path.exists()
#     saved_preds = pd.read_csv(preds_path)
#     assert saved_preds.shape[0] == predict_data.shape[0]




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
