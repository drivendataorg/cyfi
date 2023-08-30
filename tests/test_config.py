import pytest


from cyano.config import LGBParams, FeaturesConfig, ModelTrainingConfig


def test_lgbparams():
    params = LGBParams()
    assert params.early_stopping_round == 100
    assert params.num_leaves == 31

    params = LGBParams(num_leaves="12")
    assert params.num_leaves == 12


def test_features_config():
    config = FeaturesConfig()
    assert config.sample_meta_features == ["land_cover"]
    assert config.pc_days_search_window == 15

    config = FeaturesConfig(pc_meters_search_window=100, image_feature_meter_window="200")
    assert config.pc_meters_search_window == 100
    assert config.image_feature_meter_window == 200


def test_features_config_errors():
    # Unrecognized sentinel band
    with pytest.raises(ValueError):
        FeaturesConfig(use_sentinel_bands=["surprise_band"])

    # Unrecognized satellite image feature
    with pytest.raises(ValueError):
        FeaturesConfig(satellite_image_features=["surprise_feature"])

    # Unrecognized satellite meta feature
    with pytest.raises(ValueError):
        FeaturesConfig(satellite_meta_features=["surprise_feature"])

    # Unrecognized metadata feature
    with pytest.raises(ValueError):
        FeaturesConfig(sample_meta_features=["surprise_feature"])


def test_model_training_config():
    config = ModelTrainingConfig()
    assert config.n_folds == 5
    assert isinstance(config.params, LGBParams)

    config = ModelTrainingConfig(n_folds="1")
    assert config.n_folds == 1
