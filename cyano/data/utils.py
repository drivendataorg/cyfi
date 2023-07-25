from faker import Faker
import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_SEED = 40
rng = np.random.RandomState(RANDOM_SEED)
fake = Faker()
Faker.seed(RANDOM_SEED)


def add_unique_identifier(df: pd.DataFrame, id_len: int = 4) -> pd.DataFrame:
    """Given a dataframe, create a unique identifier for each row
    and set as the index

    Args:
        df (pd.DataFrame): Dataframe

    Returns:
        pd.DataFrame: Dataframe with unique identifiers as the index
    """
    df = df.copy()
    redundancy = 3
    ids = [
        fake.unique.pystr(min_chars=id_len, max_chars=id_len).lower()
        for _ in range(len(df) * redundancy)
    ]
    unique_ids = set(ids)
    while len(unique_ids) < len(df):
        redundancy += 1
        ids = [
            fake.unique.pystr(min_chars=id_len, max_chars=id_len).lower()
            for _ in range(len(df) * redundancy)
        ]
        unique_ids = set(ids)

    ids = list(unique_ids)[: len(df)]
    df["uid"] = ids

    return df.set_index("uid")
