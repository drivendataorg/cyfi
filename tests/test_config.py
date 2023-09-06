from pydantic import ValidationError
import pytest


from cyano.config import LGBParams, FeaturesConfig, ModelTrainingConfig


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


def test_features_config_cache_path(features_config):
    # Hash for caching is expected
    expected_hash = "7ab848d07eebc002140abca6f2e483a4"
    assert features_config.get_cached_path() == expected_hash

    # Same features in different order have same hash
    config1 = FeaturesConfig(use_sentinel_bands=["B01", "B02", "B03"], n_sentinel_items=3)
    config2 = FeaturesConfig(use_sentinel_bands=["B02", "B03", "B01"], n_sentinel_items=3)
    assert config1.get_cached_path() == config2.get_cached_path()

    # Irrelevant parameters don't change the hash
    config3 = FeaturesConfig(use_sentinel_bands=["B01", "B02", "B03"], n_sentinel_items=1)
    assert config1.get_cached_path() == config3.get_cached_path()

    # Different features have different hash
    config4 = FeaturesConfig(use_sentinel_bands=["B01", "B02"])
    assert config1.get_cached_path() != config4.get_cached_path()


def test_model_training_config():
    config = ModelTrainingConfig()
    assert config.n_folds == 5
    assert isinstance(config.params, LGBParams)

    config = ModelTrainingConfig(n_folds="1")
    assert config.n_folds == 1

    # Errors with extra field
    with pytest.raises(ValidationError):
        ModelTrainingConfig(extra_field="surprise_extra_field")
