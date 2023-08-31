from pydantic import ValidationError
import pytest


from cyano.config import FeaturesConfig
from cyano.experiment.experiment import ExperimentConfig


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
