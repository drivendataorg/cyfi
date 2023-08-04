from pathlib import Path
import yaml

from pydantic import BaseModel, field_serializer

from cyano.config import FeaturesConfig, ModelTrainingConfig
from cyano.pipeline import CyanoModelPipeline


class ExperimentConfig(BaseModel):
    features_config: FeaturesConfig = FeaturesConfig()
    model_training_config: ModelTrainingConfig = ModelTrainingConfig()
    train_csv: Path
    predict_csv: Path
    cache_dir: Path = None
    save_dir: Path = None

    @field_serializer("train_csv", "predict_csv", "cache_dir", "save_dir")
    def serialize_path_to_str(self, x, _info):
        return str(x)

    def run_experiment(self):
        pipeline = CyanoModelPipeline(
            features_config=self.features_config,
            model_training_config=self.model_training_config,
            cache_dir=self.cache_dir,
        )
        pipeline.run_training(train_csv=self.train_csv, save_path=self.save_dir / "model.zip")

        with open(f"{self.save_dir}/config_artifact.yaml", "w") as fp:
            yaml.dump(self.model_dump(), fp)

        pipeline.run_prediction(
            predict_csv=self.predict_csv, preds_path=self.save_dir / "preds.csv"
        )
