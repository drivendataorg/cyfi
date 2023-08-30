import pytest


from cyano.config import LGBParams, FeaturesConfig, ModelTrainingConfig


def test_lgbparams():
    params = LGBParams()
    assert params.early_stopping_round is None
    assert params.num_leaves == 31


def test_features_config():
    config = FeaturesConfig()
    assert config.metadata_features == []
    assert config.pc_days_search_window == 30


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
        FeaturesConfig(metadata_features=["surprise_feature"])


def test_model_training_config():
    config = ModelTrainingConfig()
    assert config.n_folds == 1
    assert isinstance(config.params, LGBParams)
