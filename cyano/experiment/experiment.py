from pathlib import Path
from typing import Union
import yaml

from cloudpathlib import AnyPath
import git
from loguru import logger
import pandas as pd
from pydantic import BaseModel, ConfigDict, field_serializer, field_validator

from cyano.config import FeaturesConfig, ModelTrainingConfig
from cyano.pipeline import CyanoModelPipeline
from cyano.evaluate import EvaluatePreds

REPO_ROOT = Path(__file__).parents[2].resolve()


class ExperimentConfig(BaseModel):
    features_config: FeaturesConfig = FeaturesConfig()
    model_training_config: ModelTrainingConfig = ModelTrainingConfig()
    train_csv: Union[str, Path]
    predict_csv: Union[str, Path]
    cache_dir: Path = None
    save_dir: Path = Path.cwd()
    last_commit_hash: str = None
    target_col: str = "log_density"
    debug: bool = False
    """Configuration containing parameters to be used for an end-to-end experiment

    Args:
        features_config (FeaturesConfig, optional): Features configuration. Defaults to
            FeaturesConfig().
        model_training_config (ModelTrainingConfig, optional): Model training configuration.
            Defaults to ModelTrainingConfig().
        train_csv (Union[str, Path]): Path to a training CSV with columns for date, latitude,
            longitude, and severity.
        predict_csv (Union[str, Path]): Path to a CSV for prediction and evaluation with
            columns for date, latitude, longitude, and severity.
        cache_dir (Path, optional): Cache directory. Defaults to None.
        save_dir (Path, optional): Directory to save experiment results. Defaults to
            Path.cwd().
        last_commit_hash (str, optional): Hash of the most recent commit to track codes
            used to run the experiment. Defaults to None.
        target_col (str, optional): Target column to predict. Must be either "severity",
            "density_cells_per_ml", or "log_density". Defaults to "log_density".
        debug (bool, optional): Run in debug mode. Defaults to False.
    """

    @field_validator("train_csv", "predict_csv")
    def convert_filepaths(cls, path_field):
        return AnyPath(path_field)

    # Avoid conflict with pydantic protected namespace
    model_config = ConfigDict(protected_namespaces=())

    @field_serializer("train_csv", "predict_csv", "cache_dir", "save_dir")
    def serialize_path_to_str(self, x, _info):
        return str(x)

    @classmethod
    def from_file(cls, yaml_file):
        with open(yaml_file, "r") as fp:
            config_dict = yaml.safe_load(fp)
        return cls(**config_dict)

    def run_experiment(self):
        pipeline = CyanoModelPipeline(
            features_config=self.features_config,
            model_training_config=self.model_training_config,
            cache_dir=self.cache_dir,
            target_col=self.target_col,
        )
        pipeline.run_training(
            train_csv=self.train_csv,
            save_path=self.save_dir / "model.zip",
            debug=self.debug,
        )

        # Get last commit hash to save in artifact
        repo = git.Repo(REPO_ROOT)
        self.last_commit_hash = repo.head.commit.hexsha
        with (self.save_dir / "config_artifact.yaml").open("w") as fp:
            yaml.dump(self.model_dump(), fp)
        logger.success(f"Wrote out artifact config to {self.save_dir}")

        pipeline.run_prediction(
            predict_csv=self.predict_csv, preds_path=self.save_dir / "preds.csv", debug=self.debug
        )

        # Copy train and test features to experiment dir
        for split in ["train", "test"]:
            df = pd.read_csv(pipeline.cache_dir / f"features_{split}.csv")
            df.to_csv(self.save_dir / f"features_{split}.csv", index=False)

        if self.debug:
            logger.info("Evaluation is not run in debug mode")
        else:
            EvaluatePreds(
                y_true_csv=self.predict_csv,
                y_pred_csv=self.save_dir / "preds.csv",
                save_dir=self.save_dir / "metrics",
                model_path=self.save_dir / "model.zip",
            ).calculate_all_and_save()

            logger.success(f"Wrote out metrics to {self.save_dir}/metrics")
