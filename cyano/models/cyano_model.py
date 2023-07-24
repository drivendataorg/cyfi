## Define class for ensembled set of models to predict cyanobacteria
import json
from typing import Dict, Optional

import lightgbm as lgb
from loguru import logger
import pandas as pd
from pathlib import Path

from cyano.settings import DEFAULT_LGB_PARAMS


class CyanoModel:
    def __init__(self, config: Dict, lgb_model: Optional[lgb.Booster] = None):
        """Instantiate ensembled cyanobacteria prediction model

        Args:
            config (Dict): Experiment config
            lgb_model (Optional[lgb.Booster]): LightGBM Booster model,
                if it already exists. Defaults to None.
        """
        self.config = config
        lgb_params = DEFAULT_LGB_PARAMS.copy()
        lgb_params.update(config["lgb_params"])
        self.config["lgb_params"] = lgb_params
        self.lgb_model = lgb_model

    @classmethod
    def load_model(cls, model_dir: Path) -> "CyanoModel":
        """Load an ensembled model from existing weights

        Args:
            model_dir (Path): Directory containing all model weights
                and an experiment configuration

        Returns:
            CyanoModel
        """
        # Load config from model dir
        with open(f"{model_dir}/config.json", "r") as fp:
            config = json.load(fp)

        # Load existing model
        lgb_model = lgb.Booster(model_file=f"{model_dir}/lgb_model.txt")

        # Instantiate class
        return cls(config=config, lgb_model=lgb_model)

    def save(self, save_dir: Path):
        """Save model weights and config to save_dir"""
        Path(save_dir).mkdir(exist_ok=True, parents=True)

        # Save model
        self.lgb_model.save_model(f"{save_dir}/lgb_model.txt")

        # Save config
        with open(f"{save_dir}/config.json", "w") as fp:
            json.dump(self.config, fp)

        logger.success(f"Model and config saved to {save_dir}")

    def train(self, features: pd.DataFrame, labels: pd.Series):
        """Train a cyanobacteria prediction model

        Args:
            features (pd.DataFrame): DataFrame where the index is uid
                and there is one column for each feature
            labels (pd.Series): DataFrame where the index is uid
                and there is one column for `severity`
        """
        lgb_train = lgb.Dataset(features, label=labels.loc[features.index])
        model = lgb.train(self.config["lgb_params"], lgb_train)

        self.lgb_model = model

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Predict cyanobacteria severity level

        Args:
            features (pd.DataFrame): DataFrame where the index is uid
                and there is one column for each feature

        Returns:
            pd.DataFrame: DataFrame where the index is uid and there is
                one column for predicted `severity`
        """
        if not self.lgb_model:
            raise ValueError("CyanoModel.train must be run before CyanoModel.predict")
        preds = self.lgb_model.predict(features)

        return pd.Series(data=preds, index=features.index)
