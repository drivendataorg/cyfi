from pathlib import Path
import yaml

from loguru import logger
import pandas as pd
from pydantic import BaseModel, field_serializer

from cyano.config import FeaturesConfig, ModelTrainingConfig
from cyano.pipeline import CyanoModelPipeline
from cyano.evaluate import EvaluatePreds


class ExperimentConfig(BaseModel):
    features_config: FeaturesConfig = FeaturesConfig()
    model_training_config: ModelTrainingConfig = ModelTrainingConfig()
    train_csv: Path
    predict_csv: Path
    evaluate_csv: Path
    cache_dir: Path = None
    save_dir: Path = None

    @field_serializer("train_csv", "predict_csv", "evaluate_csv", "cache_dir", "save_dir")
    def serialize_path_to_str(self, x, _info):
        return str(x)

    def run_experiment(self):
        # check indices align for evaluation
        cols = ["latitude", "longitude", "date"]
        if not (pd.read_csv(self.predict_csv)[cols] == pd.read_csv(self.evaluate_csv)[cols]).all().all():
            raise ValueError("Points in predict_csv and evaluate_csv must be the same. Check alignment of (lat, lon, date) across csvs.")

        pipeline = CyanoModelPipeline(
            features_config=self.features_config,
            model_training_config=self.model_training_config,
            cache_dir=self.cache_dir,
        )
        pipeline.run_training(train_csv=self.train_csv, save_path=self.save_dir / "model.zip")

        logger.success(f"Writing out artifact config to {self.save_dir}")
        with open(f"{self.save_dir}/config_artifact.yaml", "w") as fp:
            yaml.dump(self.model_dump(), fp)

        pipeline.run_prediction(
            predict_csv=self.predict_csv, preds_path=self.save_dir / "preds.csv"
        )

        EvaluatePreds(
            y_true_csv=self.evaluate_csv,
            y_pred_csv=self.save_dir / "preds.csv",
            save_dir=self.save_dir / "metrics"
        ).calculate_all_and_save()

        logger.success(f"Wrote out metrics to {self.save_dir}/metrics")
