from pathlib import Path
from zipfile import ZipFile

from cyano.config import FeaturesConfig, ModelTrainingConfig
from cyano.pipeline import CyanoModelPipeline

ASSETS_DIR = Path(__file__).parent / "assets"


def test_train_model_with_folds(evaluate_data_path, tmp_path):
    # Test that multiple models are trained
    n_folds = 2
    pipeline = CyanoModelPipeline(
        features_config=FeaturesConfig(),
        model_training_config=ModelTrainingConfig(n_folds=n_folds),
    )
    model_path = tmp_path / "model.zip"
    pipeline.run_training(evaluate_data_path, model_path)

    # Check how many models got saved out
    archive = ZipFile(model_path, "r")
    model_files = [name for name in archive.namelist() if "lgb_model" in name]
    assert len(model_files) == n_folds


def test_train_model_no_region(train_data_path, tmp_path):
    # Test that folds are not used when n_folds > 1 but region is missing
    pipeline = CyanoModelPipeline(
        features_config=FeaturesConfig(), model_training_config=ModelTrainingConfig(n_folds=5)
    )
    model_path = tmp_path / "model.zip"
    pipeline.run_training(train_data_path, model_path)

    # Check how many models got saved out
    archive = ZipFile(model_path, "r")
    model_files = [name for name in archive.namelist() if "lgb_model" in name]
    assert len(model_files) == 1


def test_train_model_insufficient_samples(evaluate_data_path, tmp_path):
    # Test that folds are not used when n_folds > 1 but there are insufficient samples
    pipeline = CyanoModelPipeline(
        features_config=FeaturesConfig(), model_training_config=ModelTrainingConfig(n_folds=6)
    )
    model_path = tmp_path / "model.zip"
    pipeline.run_training(evaluate_data_path, model_path)

    # Check how many models got saved out
    archive = ZipFile(model_path, "r")
    model_files = [name for name in archive.namelist() if "lgb_model" in name]
    assert len(model_files) == 1
