import hashlib

import pandas as pd


def add_unique_identifier(df: pd.DataFrame, id_len: int = 10) -> pd.DataFrame:
    """Given a dataframe with the columns []"latitude", "longitude", "date"],
    create a unique identifier for each row and set as the index

    Args:
        df (pd.DataFrame): Dataframe

    Returns:
        pd.DataFrame: Dataframe with unique identifiers as the index
    """
    df = df.copy()
    m = hashlib.md5()
    uids = []

    # create UID based on lat/lon and date
    for row in df.itertuples():
        for s in (row.latitude, row.longitude, row.date):
            m.update(str(s).encode())
        uids.append(m.hexdigest()[:id_len])

    df["uid"] = uids
    return df.set_index("uid")
