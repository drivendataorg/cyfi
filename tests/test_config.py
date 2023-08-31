from pydantic import ValidationError
import pytest


from cyano.config import LGBParams, FeaturesConfig, ModelTrainingConfig
from cyano.experiment.experiment import ExperimentConfig


def test_lgbparams():
    params = LGBParams()
    assert params.early_stopping_round == 100
    assert params.num_leaves == 31

    params = LGBParams(num_leaves="12")
    assert params.num_leaves == 12

    # Errors with extra field
    with pytest.raises(ValidationError):
        LGBParams(extra_field="surprise_extra_field")


def test_features_config():
    config = FeaturesConfig()
    assert config.sample_meta_features == ["land_cover"]
    assert config.pc_days_search_window == 15

    config = FeaturesConfig(pc_meters_search_window=100, image_feature_meter_window="200")
    assert config.pc_meters_search_window == 100
    assert config.image_feature_meter_window == 200

    # Errors with extra field
    with pytest.raises(ValidationError):
        FeaturesConfig(extra_field="surprise_extra_field")

    # Errors with unrecognized sentinel band
    with pytest.raises(ValueError):
        FeaturesConfig(use_sentinel_bands=["surprise_band"])

    # Errors with unrecognized satellite image feature
    with pytest.raises(ValueError):
        FeaturesConfig(satellite_image_features=["surprise_feature"])

    # Errors with unrecognized satellite meta feature
    with pytest.raises(ValueError):
        FeaturesConfig(satellite_meta_features=["surprise_feature"])

    # Errors with unrecognized metadata feature
    with pytest.raises(ValueError):
        FeaturesConfig(sample_meta_features=["surprise_feature"])


def test_model_training_config():
    config = ModelTrainingConfig()
    assert config.n_folds == 5
    assert isinstance(config.params, LGBParams)

    config = ModelTrainingConfig(n_folds="1")
    assert config.n_folds == 1

    # Errors with extra field
    with pytest.raises(ValidationError):
        ModelTrainingConfig(extra_field="surprise_extra_field")


def test_experiment_config(train_data_path):
    config = ExperimentConfig(
        train_csv=train_data_path,
        predict_csv=train_data_path,
        features_config=FeaturesConfig(n_sentinel_items=10),
    )
    assert config.features_config.n_sentinel_items == 10

    # Errors with extra field
    with pytest.raises(ValidationError):
        ExperimentConfig(
            train_csv=train_data_path,
            predict_csv=train_data_path,
            extra_field="surprise_extra_field",
        )
