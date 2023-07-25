## Define class for ensembled set of models to predict cyanobacteria
from typing import Dict

import pandas as pd
from pathlib import Path


class CyanoModel:
    def __init__(self, config: Dict):
        """Instantiate ensembled cyanobacteria prediction model

        Args:
            config (Dict): Model hyperparameters
        """
        self.config = config

    @classmethod
    def load_model(cls, model_dir: Path) -> "CyanoModel":
        """Load an ensembled model from existing weights

        Args:
            model_dir (Path): Directory containing all model weights
                and a configuration

        Returns:
            EnsembledModel
        """
        # Load config from model dir
        # Instantiate class
        # Load in existing weights
        pass

    def save(cls, save_dir: Path):
        """Save model weights and config to save_dir"""
        pass

    def train(self, features: pd.DataFrame, labels: pd.DataFrame):
        """Train an ensembled cyanobacteria prediction model

        Args:
            features (pd.DataFrame): DataFrame where the index is uid
                and there is one column for each feature
            labels (pd.DataFrame): DataFrame where the index is uid
                and there is one column for `severity`
        """
        pass

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Predict cyanobacteria severity level

        Args:
            features (pd.DataFrame): DataFrame where the index is uid
                and there is one column for each feature

        Returns:
            pd.DataFrame: DataFrame where the index is uid and there is
                one column for predicted `severity`
        """
        pass
