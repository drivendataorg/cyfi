import pandas as pd
from pathlib import Path
from zipfile import ZipFile

from cyfi.config import FeaturesConfig, CyFiModelConfig
from cyfi.pipeline import CyanoModelPipeline

ASSETS_DIR = Path(__file__).parent / "assets"


def test_prep_predict_data(predict_data, tmp_path):
    # Duplicates in the sample data should be dropped
    samples_path = tmp_path / "samples.csv"
    df = pd.concat([predict_data, predict_data])
    df.to_csv(samples_path, index=False)

    pipe = CyanoModelPipeline()
    pipe._prep_predict_data(samples_path)
    assert pipe.predict_samples.index.is_unique
    assert len(pipe.predict_samples) == len(predict_data)


def test_cache_dir(features_config):
    pipe = CyanoModelPipeline(
        features_config=FeaturesConfig(image_feature_meter_window=200),
        cyfi_model_config=CyFiModelConfig(num_boost_round=1000, target_col="log_density"),
    )

    # Pipeline with same features config should cache to the same place
    same_features_pipe = CyanoModelPipeline(
        features_config=FeaturesConfig(image_feature_meter_window=200),
        cyfi_model_config=CyFiModelConfig(num_boost_round=2000, target_col="severity"),
    )
    assert pipe.cache_dir == same_features_pipe.cache_dir

    # Pipeline with different features config should cache to a different place
    different_features_pipe = CyanoModelPipeline(
        features_config=FeaturesConfig(image_feature_meter_window=500),
        cyfi_model_config=CyFiModelConfig(num_boost_round=1000, target_col="log_density"),
    )
    assert pipe.cache_dir != different_features_pipe.cache_dir

    # Specified cache dir is used
    cache_dir = Path("specified_cache_dir")
    pipe_with_cache = CyanoModelPipeline(
        cache_dir=cache_dir,
        features_config=features_config,
        cyfi_model_config=CyFiModelConfig(num_boost_round=1000, target_col="log_density"),
    )
    assert cache_dir == pipe_with_cache.cache_dir.parent
    assert (cache_dir / features_config.get_cached_path()) == pipe_with_cache.cache_dir


def test_train_model_with_folds(
    evaluate_data_path, evaluate_data_features, features_config, tmp_path
):
    # Test that multiple models are trained
    # Use evaluate_data because it has a column for region
    n_folds = 2
    pipeline = CyanoModelPipeline(
        features_config=features_config,
        cyfi_model_config=CyFiModelConfig(n_folds=n_folds),
    )
    pipeline._prep_train_data(evaluate_data_path, debug=False)
    pipeline.train_features = evaluate_data_features

    model_path = tmp_path / "model.zip"
    pipeline._train_model()
    pipeline._to_disk(model_path)

    # Check how many models got saved out
    archive = ZipFile(model_path, "r")
    model_files = [name for name in archive.namelist() if "lgb_model" in name]
    assert len(model_files) == n_folds


def test_train_model_no_region(
    evaluate_data_path, evaluate_data_features, features_config, tmp_path
):
    # Test that folds are not used when n_folds > 1 but region is missing
    pipeline = CyanoModelPipeline(
        features_config=features_config, cyfi_model_config=CyFiModelConfig(n_folds=2)
    )
    pipeline._prep_train_data(evaluate_data_path, debug=False)
    pipeline.train_features = evaluate_data_features
    pipeline.train_samples = pipeline.train_samples.drop(columns=["region"])

    model_path = tmp_path / "model.zip"
    pipeline._train_model()
    pipeline._to_disk(model_path)

    # Check how many models got saved out
    archive = ZipFile(model_path, "r")
    model_files = [name for name in archive.namelist() if "lgb_model" in name]
    assert len(model_files) == 1


def test_train_model_insufficient_samples(
    evaluate_data_path, evaluate_data_features, features_config, tmp_path
):
    # Test that folds are not used when n_folds > 1 but there are insufficient samples
    pipeline = CyanoModelPipeline(
        features_config=features_config, cyfi_model_config=CyFiModelConfig(n_folds=6)
    )
    pipeline._prep_train_data(evaluate_data_path, debug=False)
    pipeline.train_features = evaluate_data_features

    model_path = tmp_path / "model.zip"
    pipeline._train_model()
    pipeline._to_disk(model_path)

    # Check how many models got saved out
    archive = ZipFile(model_path, "r")
    model_files = [name for name in archive.namelist() if "lgb_model" in name]
    assert len(model_files) == 1
