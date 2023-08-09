from pathlib import Path
import yaml

from loguru import logger
from pydantic import BaseModel, field_serializer

from cyano.config import FeaturesConfig, ModelTrainingConfig
from cyano.pipeline import CyanoModelPipeline
from cyano.evaluate import EvaluatePreds


class ExperimentConfig(BaseModel):
    features_config: FeaturesConfig = FeaturesConfig()
    model_training_config: ModelTrainingConfig = ModelTrainingConfig()
    train_csv: Path
    predict_csv: Path
    cache_dir: Path = None
    save_dir: Path = None
    debug: bool = False

    @field_serializer("train_csv", "predict_csv", "cache_dir", "save_dir")
    def serialize_path_to_str(self, x, _info):
        return str(x)

    def run_experiment(self):
        pipeline = CyanoModelPipeline(
            features_config=self.features_config,
            model_training_config=self.model_training_config,
            cache_dir=self.cache_dir,
        )
        pipeline.run_training(
            train_csv=self.train_csv, save_path=self.save_dir / "model.zip", debug=self.debug
        )

        logger.success(f"Writing out artifact config to {self.save_dir}")
        with open(f"{self.save_dir}/config_artifact.yaml", "w") as fp:
            yaml.dump(self.model_dump(), fp)

        pipeline.run_prediction(
            predict_csv=self.predict_csv, preds_path=self.save_dir / "preds.csv", debug=self.debug
        )

        if self.debug:
            logger.info("Evaluation is not run in debug mode")
        else:
            EvaluatePreds(
                y_true_csv=self.predict_csv,
                y_pred_csv=self.save_dir / "preds.csv",
                save_dir=self.save_dir / "metrics",
            ).calculate_all_and_save()

            logger.success(f"Wrote out metrics to {self.save_dir}/metrics")
