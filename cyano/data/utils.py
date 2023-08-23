import hashlib

import pandas as pd

from cyano.settings import SEVERITY_LEFT_EDGES


def add_unique_identifier(df: pd.DataFrame) -> pd.DataFrame:
    """Given a dataframe with the columns []"latitude", "longitude", "date"],
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


def convert_density_to_severity(
    df: pd.DataFrame, density_col_name: str = "severity"
) -> pd.DataFrame:
    """Convert exact density to binned severity

    Args:
        df (pd.DataFrame): Dataframe with a column for exact density
        density_col_name (str, optional): Name of the columns with
            exact density in cells/mL

    Returns:
        pd.DataFrame: Dataframe with a column for severity
            instead of exact density
    """
    df["density"] = df[density_col_name].copy()
    df = df.drop(columns=[density_col_name])

    df["severity"] = pd.cut(
        df.density,
        SEVERITY_LEFT_EDGES + [SEVERITY_LEFT_EDGES[-1] * 2],
        include_lowest=True,
        right=False,
        labels=range(1, 6),
    ).astype(float)

    # Fill in values higher than max value
    df.loc[df.density >= SEVERITY_LEFT_EDGES[-1], "severity"] = 5

    # Fill in negative density preds with severity 1
    df.loc[df.density <= 0, "severity"] = 1

    return df
