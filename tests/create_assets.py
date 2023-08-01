import yaml
from pathlib import Path

from loguru import logger
import pandas as pd

from cyano.config import TrainConfig, PredictConfig, ModelConfig, FeaturesConfig
from cyano.model_manager import train_model

ASSETS_DIR = Path(__file__).parent / "assets"


def main():
    """Helper script to train a model using the default TrainConfig and then write out
    the following test assets:
    - train_config.yaml
    - trained_model directory
    - predict_config.yaml
    """
    train_data = pd.read_csv(ASSETS_DIR / "train_data.csv")

    # generally use defaults in TrainConfig
    train_config = TrainConfig(
        features_config=FeaturesConfig(cache_dir="/tmp/feature_cache"),
        tree_model_config=ModelConfig(save_dir="tests/assets/trained_model"),
    )
    train_model(train_data, train_config)

    # write out to assets directory
    with (ASSETS_DIR / "train_config.yaml").open("w") as f:
        yaml.dump(train_config.model_dump(), f)
        logger.success("Wrote out train_config.yaml to assets directory")

    # create predict config that uses trained model
    predict_dict = train_config.sanitize()
    predict_dict["preds_path"] = "preds.csv"

    with (ASSETS_DIR / "predict_config.yaml").open("w") as f:
        yaml.dump(PredictConfig(**predict_dict).model_dump(), f)
        logger.success("Wrote out predict_config.yaml to assets directory")


if __name__ == "__main__":
    main()
