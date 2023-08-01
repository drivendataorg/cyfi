## Define class for ensembled set of models to predict cyanobacteria
from typing import Optional

import lightgbm as lgb
from loguru import logger
import pandas as pd
from pathlib import Path

from cyano.config import ModelConfig


class CyanoModel:
    def __init__(self, config: ModelConfig, lgb_model: Optional[lgb.Booster] = None):
        """Instantiate ensembled cyanobacteria prediction model

        Args:
            config (ModelConfig): Model config
            lgb_model (Optional[lgb.Booster]): LightGBM Booster model,
                if it already exists. Defaults to None.
        """
        self.config = config
        self.lgb_model = lgb_model

    @classmethod
    def load_model(cls, config: ModelConfig) -> "CyanoModel":
        """Load an ensembled model from existing weights

        Args:
            config (ModelConfig): Model config

        Returns:
            CyanoModel
        """
        # Load existing model
        lgb_model = lgb.Booster(model_file=f"{config.save_dir}/lgb_model.txt")

        # Instantiate class
        return cls(config=config, lgb_model=lgb_model)

    def save(self, save_dir: Path):
        """Save model weights and config to save_dir"""
        Path(save_dir).mkdir(exist_ok=True, parents=True)

        # Save model
        self.lgb_model.save_model(f"{save_dir}/lgb_model.txt")
        logger.success(f"Model weights saved to {save_dir}")

    def train(self, features: pd.DataFrame, labels: pd.Series):
        """Train a cyanobacteria prediction model

        Args:
            features (pd.DataFrame): DataFrame where the index is uid
                and there is one column for each feature
            labels (pd.Series): DataFrame where the index is uid
                and there is one column for `severity`
        """
        lgb_train = lgb.Dataset(features, label=labels.loc[features.index])
        model = lgb.train(
            self.config.params.model_dump(),
            lgb_train,
            num_boost_round=self.config.num_boost_round,
        )

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
