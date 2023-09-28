from pathlib import Path
import shutil
import sys
from typing import Union
import yaml

from cloudpathlib import AnyPath
from dotenv import load_dotenv, find_dotenv
import git
from loguru import logger
from pydantic import BaseModel, ConfigDict, field_serializer, field_validator
import typer

from cyfi.config import FeaturesConfig, CyFiModelConfig
from cyfi.pipeline import CyFiPipeline
from cyfi.evaluate import EvaluatePreds

REPO_ROOT = Path(__file__).parents[1].resolve()

app = typer.Typer(pretty_exceptions_show_locals=False)

load_dotenv(find_dotenv())

# Set logger to only log info or higher
logger.remove()
logger.add(sys.stderr, level="INFO")


class ExperimentConfig(BaseModel):
    """Configuration containing parameters to be used for an end-to-end experiment

    Args:
        train_csv (Union[str, Path]): Path to a training CSV with columns for date, latitude,
            longitude, and severity. Latitude and longitude must be in coordinate reference
            system WGS-84 (EPSG:4326).
        predict_csv (Union[str, Path]): Path to a CSV for prediction and evaluation with
            columns for date, latitude, longitude, and severity.
        features_config (FeaturesConfig, optional): Features configuration. Defaults to
            FeaturesConfig().
        cyfi_model_config (CyFiModelConfig, optional): Model configuration. Defaults to CyFiModelConfig().
        cache_dir (Path, optional): Cache directory. Defaults to None.
        save_dir (Path, optional): Directory to save experiment results. Defaults to
            Path.cwd().
        last_commit_hash (str, optional): Hash of the most recent commit to track codes
            used to run the experiment. Defaults to None.
        debug (bool, optional): Run in debug mode. Defaults to False.
    """

    train_csv: Union[str, Path]
    predict_csv: Union[str, Path]
    features_config: FeaturesConfig = FeaturesConfig()
    cyfi_model_config: CyFiModelConfig = CyFiModelConfig()
    cache_dir: Path = None
    save_dir: Path = Path.cwd()
    last_commit_hash: str = None
    debug: bool = False

    # Do not allow extra fields and silence warning for conflict with pydantic protected namespace
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    @field_validator("train_csv", "predict_csv")
    def convert_filepaths(cls, path_field):
        return AnyPath(path_field)

    @field_serializer("train_csv", "predict_csv", "cache_dir", "save_dir")
    def serialize_path_to_str(self, x, _info):
        return str(x)

    @classmethod
    def from_file(cls, yaml_file):
        with open(yaml_file, "r") as fp:
            config_dict = yaml.safe_load(fp)
        return cls(**config_dict)

    def run_experiment(self):
        pipeline = CyFiPipeline(
            features_config=self.features_config,
            cyfi_model_config=self.cyfi_model_config,
            cache_dir=self.cache_dir,
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

        # Copy train and test features and Sentinel metadata to experiment dir
        for split in ["train", "test"]:
            shutil.copy(
                pipeline.cache_dir / f"features_{split}.csv",
                self.save_dir / f"features_{split}.csv",
            )
            shutil.copy(
                pipeline.cache_dir / f"sentinel_metadata_{split}.csv",
                self.save_dir / f"sentinel_metadata_{split}.csv",
            )

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


@app.command()
def run_experiment(
    config_path: Path = typer.Argument(exists=True, help="Path to an experiment configuration")
):
    """Run an experiment"""
    config = ExperimentConfig.from_file(config_path)
    logger.add(config.save_dir / "experiment.log", level="DEBUG")
    config.run_experiment()


if __name__ == "__main__":
    app()
