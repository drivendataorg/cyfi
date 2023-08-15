from pathlib import Path
from typing import Union
import yaml

from cloudpathlib import AnyPath
import git
from loguru import logger
from pydantic import BaseModel, ConfigDict, field_serializer, field_validator

from cyano.config import FeaturesConfig, ModelTrainingConfig
from cyano.pipeline import CyanoModelPipeline
from cyano.evaluate import EvaluatePreds
from cyano.settings import REPO_ROOT


class ExperimentConfig(BaseModel):
    features_config: FeaturesConfig = FeaturesConfig()
    model_training_config: ModelTrainingConfig = ModelTrainingConfig()
    train_csv: Union[str, Path]
    predict_csv: Union[str, Path]
    cache_dir: Path = None
    save_dir: Path = Path.cwd()
    last_commit_hash: str = None
    debug: bool = False

    @field_validator("train_csv", "predict_csv")
    def convert_filepaths(cls, path_field):
        return AnyPath(path_field)

    # Avoid conflict with pydantic protected namespace
    model_config = ConfigDict(protected_namespaces=())

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

        # Get last commit hash to save in artifact
        repo = git.Repo(REPO_ROOT.parent)
        self.last_commit_hash = repo.head.commit.hexsha
        with (self.save_dir / "config_artifact.yaml").open("w") as fp:
            yaml.dump(self.model_dump(), fp)
        logger.success(f"Wrote out artifact config to {self.save_dir}")

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
                model=pipeline.model,
            ).calculate_all_and_save()

            logger.success(f"Wrote out metrics to {self.save_dir}/metrics")
