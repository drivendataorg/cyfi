import numpy as np

from cyano.data.utils import (
    add_unique_identifier,
    convert_density_to_log_density,
    convert_log_density_to_density,
)


def test_add_unique_identifier(train_data):
    uids = add_unique_identifier(train_data).index.values
    assert (
        uids
        == [
            "e3ebefd90a00c3cc9f5aeaf32cd4c184",
            "671520fa92f555ab335e0cfa888c57e7",
            "9c601f226c2af07d570134127a7fda27",
            "3a2c48812b551d720f8d56772efa6df1",
            "2543db364f727f17fe4ce7881aa180da",
        ]
    ).all()


def test_convert_density_to_log_density(train_data):
    converted_log_density = convert_density_to_log_density(train_data.density_cells_per_ml)
    assert np.isclose(converted_log_density.iloc[0], 15.584939377137637)


def test_convert_log_density_to_density(train_data):
    converted_density = convert_log_density_to_density(train_data.log_density)
    assert np.isclose(converted_density.iloc[0], 5867499.999999985)
