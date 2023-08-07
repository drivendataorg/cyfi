from cyano.data.utils import add_unique_identifier


def test_add_unique_identifier(train_data):
    uids = add_unique_identifier(train_data).index.values
    assert (uids == ["e3ebefd90a", "671520fa92", "9c601f226c", "3a2c48812b", "2543db364f"]).all()
