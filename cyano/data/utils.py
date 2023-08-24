import hashlib

import numpy as np
import pandas as pd

from cyano.settings import SEVERITY_LEFT_EDGES


def add_unique_identifier(df: pd.DataFrame) -> pd.DataFrame:
    """Given a dataframe with the columns ["latitude", "longitude", "date"],
    create a unique identifier for each row and set as the index

    Args:
        df (pd.DataFrame): Dataframe

    Returns:
        pd.DataFrame: Dataframe with unique identifiers as the index
    """
    df = df.copy()
    uids = []

    # create UID based on lat/lon and date
    for row in df.itertuples():
        m = hashlib.md5()
        for s in (row.latitude, row.longitude, row.date):
            m.update(str(s).encode())
        uids.append(m.hexdigest())

    df["sample_id"] = uids
    return df.set_index("sample_id")


def convert_density_to_severity(density_series: pd.Series) -> pd.Series:
    """Convert exact density to binned severity

    Args:
        density_series (pd.Series): Series containing density values

    Returns:
        pd.Series: Series containing severity buckets
    """
    density = pd.cut(
        density_series,
        SEVERITY_LEFT_EDGES + [np.inf],
        include_lowest=True,
        right=False,
        labels=np.arange(1, 6),
    ).astype(float)

    return density
