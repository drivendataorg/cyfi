from cyano.data.utils import add_unique_identifier


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
